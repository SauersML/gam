//! Root-cause regression for gam#903: the `bs="fs"` factor smooth is a RANDOM
//! EFFECT, so each group's per-group line must shrink toward the POPULATION
//! trend (the mixed-model BLUP), not toward zero. The earlier per-dimension null
//! ridge `I_L ⊗ z zᵀ` shrank every group's slope toward zero, which biases the
//! population slope toward zero and ruins held-out extrapolation. This test pins
//! the corrected behaviour from a different angle than the sleepstudy forecast:
//! on SPARSE, short-range training data it asserts that partial pooling toward
//! the population slope actually helps, by beating the no-pooling per-group OLS
//! on an EXTRAPOLATED hold-out — something a shrink-to-zero fit cannot do (it is
//! strictly worse than OLS out of sample). No reference tool: the baselines are
//! computed in-process, so this runs anywhere.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 14;
const N_TRAIN_PER_GROUP: usize = 5; // sparse: 5 short-range points per group
const N_TEST_PER_GROUP: usize = 4;
const POP_INTERCEPT: f64 = 1.0;
const POP_SLOPE: f64 = 4.0; // the population trend the random slopes vary around
const SLOPE_SD: f64 = 1.0; // genuine between-group slope spread
const INTERCEPT_SD: f64 = 2.0;
const NOISE_SD: f64 = 0.5;
const SEED: u64 = 9_1_7;

/// Training x live in a short interval; the held-out x are an extrapolation
/// beyond it, where the shrinkage TARGET of the per-group slope dominates the
/// forecast.
const TRAIN_XLO: f64 = 0.0;
const TRAIN_XHI: f64 = 0.4;
const TEST_X: [f64; N_TEST_PER_GROUP] = [0.7, 0.8, 0.9, 1.0];

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Ordinary least squares slope+intercept of `y` on `x`.
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
fn fs_random_slope_shrinks_toward_population_not_zero() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(TRAIN_XLO, TRAIN_XHI).expect("uniform");
    let slope_dev = Normal::new(0.0, SLOPE_SD).expect("slope normal");
    let int_dev = Normal::new(0.0, INTERCEPT_SD).expect("intercept normal");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise normal");

    // True per-group line: population trend + group deviation.
    let group_slope: Vec<f64> = (0..N_GROUPS)
        .map(|_| POP_SLOPE + slope_dev.sample(&mut rng))
        .collect();
    let group_intercept: Vec<f64> = (0..N_GROUPS)
        .map(|_| POP_INTERCEPT + int_dev.sample(&mut rng))
        .collect();

    // ---- build the sparse training set --------------------------------------
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

    // ---- held-out extrapolation targets (noiseless truth) -------------------
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

    // ---- fit gam: y ~ fs(x, g) ---------------------------------------------
    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, train_rows).expect("encode fs train");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ fs(x, g)", &ds, &cfg).expect("gam fs fit")
    else {
        panic!("expected a standard GAM fit");
    };

    // gam predictions at the held-out (group, x) grid.
    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (row, (&gi, &xe)) in test_group.iter().zip(&test_x).enumerate() {
        grid[[row, x_idx]] = xe;
        grid[[row, g_idx]] = gi as f64;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild fs design");
    let gam_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // Recover gam's per-group fitted slope by finite difference of the fitted
    // surface across the held-out x (a clean linear extrapolation region).
    let mut gam_group_slope = vec![0.0; N_GROUPS];
    for gi in 0..N_GROUPS {
        let lo = gi * N_TEST_PER_GROUP;
        let hi = lo + N_TEST_PER_GROUP;
        let (s, _) = ols(&test_x[lo..hi], &gam_pred[lo..hi]);
        gam_group_slope[gi] = s;
    }
    let mean_gam_slope = gam_group_slope.iter().sum::<f64>() / N_GROUPS as f64;

    // ---- baselines computed in-process -------------------------------------
    // (a) NO POOLING: each group's own OLS extrapolated to the test x.
    // (b) FULL POOLING: a single population OLS on all training rows.
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
        "[fs-blup] groups={N_GROUPS} train/group={N_TRAIN_PER_GROUP} \
         true_pop_slope={POP_SLOPE:.2} gam_mean_slope={mean_gam_slope:.3} \
         RMSE gam={gam_rmse:.3} no-pool-OLS={nopool_rmse:.3} full-pool-OLS={fullpool_rmse:.3}"
    );

    // (1) The population slope is RECOVERED, not shrunk to zero. The legacy
    // shrink-to-zero ridge drove the mean fitted slope toward 0; the corrected
    // construction keeps it near the true population slope.
    assert!(
        mean_gam_slope >= 0.55 * POP_SLOPE,
        "fs mean group slope collapsed toward zero: {mean_gam_slope:.3} < {:.3} \
         (= 0.55 * true pop slope {POP_SLOPE:.2}) — population trend not recovered",
        0.55 * POP_SLOPE
    );

    // (2) Partial pooling toward the population slope HELPS out of sample: gam
    // beats the no-pooling per-group OLS on the extrapolated hold-out. A
    // shrink-to-zero fit is strictly worse than OLS here, so this fails under
    // the old construction and passes under the BLUP-correct one.
    assert!(
        gam_rmse <= nopool_rmse,
        "fs did not beat no-pooling OLS out of sample: gam={gam_rmse:.3} > no-pool={nopool_rmse:.3} \
         (partial pooling toward the population slope should reduce extrapolation error)"
    );

    // (3) Sanity: gam genuinely uses the group structure — it is well clear of
    // the group-blind full-pooling forecast.
    assert!(
        gam_rmse < fullpool_rmse,
        "fs no better than group-blind full pooling: gam={gam_rmse:.3} >= full-pool={fullpool_rmse:.3}"
    );
}
