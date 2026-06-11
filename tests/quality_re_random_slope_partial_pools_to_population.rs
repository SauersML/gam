//! Root-cause regression for gam#903: gam's `bs="re"` random slope must be a
//! PARAMETRIC random intercept+slope `[1, x]` (mgcv's `(1 + x | g)`), shrinking
//! each group toward the FIXED population trend — not a piecewise-linear B-spline
//! with the pooled knot heuristic, which over-parameterized the term (≈6 wiggly
//! coefficients/group), ill-conditioned the REML/joint-Newton solve (minute-long
//! fits) and broke the partial pooling.
//!
//! This pins the corrected behaviour with NO reference tool: on SPARSE,
//! short-range training data it asserts that partial pooling toward the
//! population slope actually helps, by beating the no-pooling per-group OLS on an
//! EXTRAPOLATED hold-out — something the over-parameterized / shrink-to-its-own-
//! noise fit cannot do. It also checks the fit is FAST (the over-parameterized
//! term took minutes) and that the per-group basis is the 2-coefficient linear
//! random effect, not a multi-knot smooth.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 16;
const N_TRAIN_PER_GROUP: usize = 5; // sparse, short-range
const N_TEST_PER_GROUP: usize = 4;
const POP_INTERCEPT: f64 = 1.0;
const POP_SLOPE: f64 = 4.0;
const SLOPE_SD: f64 = 1.0;
const INTERCEPT_SD: f64 = 2.0;
const NOISE_SD: f64 = 0.5;
const SEED: u64 = 90_3_42;
const TRAIN_XLO: f64 = 0.0;
const TRAIN_XHI: f64 = 0.4;
const TEST_X: [f64; N_TEST_PER_GROUP] = [0.7, 0.8, 0.9, 1.0];

/// OLS slope+intercept of `y` on `x`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let xbar = x.iter().sum::<f64>() / n;
    let ybar = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|&v| (v - xbar) * (v - xbar)).sum();
    let sxy: f64 = x
        .iter()
        .zip(y)
        .map(|(&xv, &yv)| (xv - xbar) * (yv - ybar))
        .sum();
    let slope = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    (slope, ybar - slope * xbar)
}

#[test]
fn re_random_slope_partial_pools_toward_population() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(TRAIN_XLO, TRAIN_XHI).expect("uniform");
    let slope_dev = Normal::new(0.0, SLOPE_SD).expect("slope normal");
    let int_dev = Normal::new(0.0, INTERCEPT_SD).expect("intercept normal");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise normal");

    let group_slope: Vec<f64> = (0..N_GROUPS)
        .map(|_| POP_SLOPE + slope_dev.sample(&mut rng))
        .collect();
    let group_intercept: Vec<f64> = (0..N_GROUPS)
        .map(|_| POP_INTERCEPT + int_dev.sample(&mut rng))
        .collect();

    let mut train_rows = Vec::<StringRecord>::new();
    let mut train_x_by_group: Vec<Vec<f64>> = vec![Vec::new(); N_GROUPS];
    let mut train_y_by_group: Vec<Vec<f64>> = vec![Vec::new(); N_GROUPS];
    for gi in 0..N_GROUPS {
        for _ in 0..N_TRAIN_PER_GROUP {
            let xi = ux.sample(&mut rng);
            let yi = group_intercept[gi] + group_slope[gi] * xi + noise.sample(&mut rng);
            train_rows.push(StringRecord::from(vec![
                format!("{xi:.17e}"),
                format!("g{gi}"),
                format!("{yi:.17e}"),
            ]));
            train_x_by_group[gi].push(xi);
            train_y_by_group[gi].push(yi);
        }
    }

    let mut test_group = Vec::<usize>::new();
    let mut test_x = Vec::<f64>::new();
    let mut test_truth = Vec::<f64>::new();
    for gi in 0..N_GROUPS {
        for &xe in TEST_X.iter() {
            test_group.push(gi);
            test_x.push(xe);
            test_truth.push(group_intercept[gi] + group_slope[gi] * xe);
        }
    }
    let n_test = test_truth.len();

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, train_rows).expect("encode re train");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // gam's random slope: fixed x + parametric per-group random intercept+slope.
    let t0 = std::time::Instant::now();
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x + s(x, g, bs=\"re\")", &ds, &cfg).expect("gam re fit")
    else {
        panic!("expected a standard GAM fit");
    };
    let fit_secs = t0.elapsed().as_secs_f64();
    let gam_edf = fit.fit.edf_total().expect("edf");

    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (row, (&gi, &xe)) in test_group.iter().zip(&test_x).enumerate() {
        grid[[row, x_idx]] = xe;
        grid[[row, g_idx]] = gi as f64;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild re design");
    let gam_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // Per-group fitted slope (linear region).
    let mut gam_group_slope = vec![0.0; N_GROUPS];
    for gi in 0..N_GROUPS {
        let lo = gi * N_TEST_PER_GROUP;
        let (s, _) = ols(
            &test_x[lo..lo + N_TEST_PER_GROUP],
            &gam_pred[lo..lo + N_TEST_PER_GROUP],
        );
        gam_group_slope[gi] = s;
    }
    let mean_gam_slope = gam_group_slope.iter().sum::<f64>() / N_GROUPS as f64;

    // No-pooling and full-pooling OLS baselines.
    let mut all_x = Vec::new();
    let mut all_y = Vec::new();
    for gi in 0..N_GROUPS {
        all_x.extend_from_slice(&train_x_by_group[gi]);
        all_y.extend_from_slice(&train_y_by_group[gi]);
    }
    let (pop_slope, pop_int) = ols(&all_x, &all_y);
    let mut nopool_pred = vec![0.0; n_test];
    let mut fullpool_pred = vec![0.0; n_test];
    for (row, (&gi, &xe)) in test_group.iter().zip(&test_x).enumerate() {
        let (gs, gint) = ols(&train_x_by_group[gi], &train_y_by_group[gi]);
        nopool_pred[row] = gint + gs * xe;
        fullpool_pred[row] = pop_int + pop_slope * xe;
    }

    let gam_rmse = rmse(&gam_pred, &test_truth);
    let nopool_rmse = rmse(&nopool_pred, &test_truth);
    let fullpool_rmse = rmse(&fullpool_pred, &test_truth);

    eprintln!(
        "[re-blup] groups={N_GROUPS} train/group={N_TRAIN_PER_GROUP} edf={gam_edf:.2} \
         fit_secs={fit_secs:.2} true_pop_slope={POP_SLOPE:.2} gam_mean_slope={mean_gam_slope:.3} \
         RMSE gam={gam_rmse:.3} no-pool-OLS={nopool_rmse:.3} full-pool-OLS={fullpool_rmse:.3}"
    );

    // (0) The fit is fast and the random-effect term is the 2-coefficient linear
    // random effect, not an over-parameterized multi-knot smooth. A fixed
    // intercept+slope (2) plus a 16-group [1,x] random effect (32) is ~34 columns;
    // its edf must sit well below that, and far below the ~100 a 6-function/group
    // piecewise-linear basis would imply.
    assert!(
        fit_secs < 30.0,
        "re random-slope fit too slow ({fit_secs:.1}s) — over-parameterized term regressed"
    );
    assert!(
        gam_edf < 34.0,
        "re effective dof {gam_edf:.2} implies a multi-knot smooth, not a [1,x] random effect"
    );

    // (1) Population slope recovered, not collapsed toward zero.
    assert!(
        mean_gam_slope >= 0.6 * POP_SLOPE && mean_gam_slope <= 1.4 * POP_SLOPE,
        "re mean group slope off the population slope: {mean_gam_slope:.3} vs true {POP_SLOPE:.2}"
    );

    // (2) Partial pooling toward the population slope HELPS out of sample: gam
    // beats no-pooling per-group OLS on the extrapolated hold-out. The
    // over-parameterized / shrink-to-noise fit cannot.
    assert!(
        gam_rmse <= nopool_rmse,
        "re did not beat no-pooling OLS out of sample: gam={gam_rmse:.3} > no-pool={nopool_rmse:.3}"
    );

    // (3) Sanity: gam genuinely uses the group structure (clear of group-blind).
    assert!(
        gam_rmse < fullpool_rmse,
        "re no better than group-blind full pooling: gam={gam_rmse:.3} >= full-pool={fullpool_rmse:.3}"
    );
}
