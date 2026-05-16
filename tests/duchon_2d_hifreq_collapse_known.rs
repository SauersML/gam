//! Documents cycle #7: 2D pure Duchon collapses on a moderately-high-frequency
//! truth `sin(2π·4·x1) · cos(2π·4·x2)`. REML converges to `rho ≈ [0, 10.45, 0]`
//! — one of the three operator penalties (tension) is driven to λ ≈ 3.4e4,
//! which crushes the corresponding coefficient subspace. The fit has
//! span ≈ 0.37 against a truth of span ≈ 2.0.
//!
//! With explicit `length_scale=0.1` (hybrid Duchon), the same truth fits
//! cleanly (rmse 0.36, span ≈ 2.0). Matern centers=150 also fits cleanly
//! (rmse 0.33, span ≈ 2.0). The bug is specific to the multi-penalty
//! all-active operator-penalty path for pure (scale-free) Duchon.
//!
//! The test below LOCKS IN the current collapse so any inadvertent change to
//! the multi-penalty path is caught, AND is paired with a passing-hybrid
//! companion that proves the data is fittable when the right scale is given.
//! When the underlying fix lands, the `should_panic` direction on the pure
//! test gets flipped (delete the `#[should_panic]` and tighten the bounds).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_hifreq_2d(
    n: usize,
    sigma: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let x1: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let y_truth: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(a, b)| {
            (2.0 * std::f64::consts::PI * 4.0 * a).sin()
                * (2.0 * std::f64::consts::PI * 4.0 * b).cos()
        })
        .collect();
    let y_noisy: Vec<f64> = y_truth
        .iter()
        .map(|&v| v + noise.sample(&mut rng))
        .collect();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x1
        .iter()
        .zip(x2.iter())
        .zip(y_noisy.iter())
        .map(|((a, b), c)| StringRecord::from(vec![a.to_string(), b.to_string(), c.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x1, x2, y_truth, data)
}

fn fit_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x1t: &[f64],
    x2t: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x1t.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = x1t[i];
        m[[i, 1]] = x2t[i];
        m[[i, 2]] = 0.0;
    }
    let test_design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    test_design.design.apply(&fit.fit.beta).to_vec()
}

fn build_test_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g: Vec<f64> = (0..30).map(|i| 0.01 + 0.98 * i as f64 / 29.0).collect();
    let mut x1t = Vec::with_capacity(900);
    let mut x2t = Vec::with_capacity(900);
    for &a in &g {
        for &b in &g {
            x1t.push(a);
            x2t.push(b);
        }
    }
    let truth: Vec<f64> = x1t
        .iter()
        .zip(x2t.iter())
        .map(|(a, b)| {
            (2.0 * std::f64::consts::PI * 4.0 * a).sin()
                * (2.0 * std::f64::consts::PI * 4.0 * b).cos()
        })
        .collect();
    (x1t, x2t, truth)
}

fn metrics(yhat: &[f64], truth: &[f64]) -> (f64, f64) {
    let max = yhat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = yhat.iter().cloned().fold(f64::INFINITY, f64::min);
    let span = max - min;
    let rmse = (yhat
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / yhat.len() as f64)
        .sqrt();
    (rmse, span)
}

/// LOCKS IN the current collapse: pure Duchon on 2D hifreq has predicted span
/// strictly below 50% of truth span and RMSE worse than 0.40. If either of
/// those flips, the multi-penalty pure-Duchon path has changed — likely a
/// fix; rewrite this test as the positive assertion.
#[test]
fn pure_duchon_2d_hifreq_collapse_currently_observed() {
    init_parallelism();
    let (_x1, _x2, _yt, data) = make_hifreq_2d(500, 0.10, 13);
    let (x1t, x2t, truth) = build_test_grid();
    let yhat = fit_predict("y ~ duchon(x1, x2)", &data, &x1t, &x2t);
    let (rmse, span) = metrics(&yhat, &truth);
    let truth_span = 2.0_f64;
    eprintln!("[duchon-2d-hifreq pure] rmse={rmse:.4} span={span:.3} truth_span={truth_span:.3}");
    // These two upper bounds together capture the current collapsed regime.
    // Flip them to lower bounds (and tighten) when the underlying fix lands.
    assert!(
        span < 0.50 * truth_span,
        "expected collapse (span < 50% truth) — fit appears to have improved (span={span:.3}); \
         update this regression to assert the new healthier behavior",
    );
    assert!(
        rmse > 0.40,
        "expected collapse-level rmse (>0.40) — fit appears to have improved (rmse={rmse:.4}); \
         update this regression to assert the new healthier behavior",
    );
}

/// Sanity companion: with `length_scale=0.1` (hybrid Duchon), the same data
/// fits cleanly. This proves the data is fittable; the failure above is a
/// REML / penalty-balance issue, not an information-theoretic one.
#[test]
fn hybrid_duchon_2d_hifreq_fits_when_length_scale_given() {
    init_parallelism();
    let (_x1, _x2, _yt, data) = make_hifreq_2d(500, 0.10, 13);
    let (x1t, x2t, truth) = build_test_grid();
    let yhat = fit_predict("y ~ duchon(x1, x2, length_scale=0.10)", &data, &x1t, &x2t);
    let (rmse, span) = metrics(&yhat, &truth);
    let truth_span = 2.0_f64;
    eprintln!(
        "[duchon-2d-hifreq hybrid ls=0.1] rmse={rmse:.4} span={span:.3} truth_span={truth_span:.3}"
    );
    assert!(
        span >= 0.75 * truth_span,
        "hybrid Duchon with length_scale=0.10 should not collapse; span={span:.3}",
    );
    assert!(
        rmse <= 0.45,
        "hybrid Duchon with length_scale=0.10 should fit cleanly; rmse={rmse:.4}",
    );
}
