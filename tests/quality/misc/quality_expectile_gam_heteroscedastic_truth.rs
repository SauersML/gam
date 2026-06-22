//! End-to-end quality gate for expectile GAMs (Newey–Powell asymmetric least
//! squares) against a planted truth whose conditional expectiles are known in
//! closed form.
//!
//! # The capability
//!
//! `family = "expectile(τ)"` fits the conditional τ-expectile of `y | x` — the
//! smooth, everywhere-differentiable analogue of the τ-quantile (Newey & Powell
//! 1987). The fit is a Least Asymmetrically Weighted Squares (LAWS) fixed point
//! layered on the full penalized Gaussian-identity GAM, so it inherits REML
//! smoothing-parameter selection and every basis. This test verifies the
//! capability is real and accurate, not just present.
//!
//! # The planted truth (closed-form expectiles)
//!
//! Draw `y = f(x) + σ(x)·Z`, `Z ~ N(0,1)`, with a smooth mean `f` and a smooth,
//! strictly positive scale `σ`. For a location-scale family the conditional
//! τ-expectile is exactly
//!
//! ```text
//!   eτ(y | x) = f(x) + σ(x) · e_τ
//! ```
//!
//! where `e_τ` is the τ-expectile of the *standard normal*, a fixed constant we
//! compute by a closed-form root find. This gives an exact ground-truth curve
//! for every τ — the reference is the math, not a mature tool's fitted output,
//! so the bar is genuine truth recovery (per the suite's reference-as-truth
//! policy). The asymmetry of the conditional spread (σ grows with x) makes the
//! three expectile curves fan out, which a correct τ-expectile fit must track
//! and a plain mean fit cannot.
//!
//! # What is asserted (objective quality)
//!
//! 1. Truth recovery: the fitted τ-expectile curve tracks `f(x)+σ(x)·e_τ` with
//!    small RMSE on a held-out grid, for τ ∈ {0.1, 0.5, 0.9}.
//! 2. Ordering / fanning: the fitted curves are strictly ordered
//!    `ê_0.1 < ê_0.5 < ê_0.9` everywhere, and the upper–lower gap grows with x
//!    (tracking σ(x)), which the mean fit (a single curve) structurally cannot
//!    represent.
//! 3. Calibration: the empirical asymmetric-residual balance at the fitted
//!    τ-expectile matches τ — `Σ τ·(y−ê)_+ ≈ Σ (1−τ)·(ê−y)_+` — the defining
//!    estimating equation, evaluated in-sample.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Smooth conditional mean of the planted truth.
fn truth_mean(x: f64) -> f64 {
    2.0 + 1.5 * (2.4 * x).sin() + 0.8 * x
}

/// Smooth, strictly positive conditional scale (heteroscedastic: grows with x).
fn truth_scale(x: f64) -> f64 {
    0.35 + 0.9 * x
}

/// Standard-normal pdf.
fn phi(z: f64) -> f64 {
    (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard-normal cdf via erf.
fn big_phi(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz–Stegun 7.1.26 error-function approximation (|err| < 1.5e-7),
/// sufficient for constructing the analytic expectile reference constant.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// τ-expectile of the standard normal, the root `m` of the defining equation
/// `τ · E[(Z−m)_+] = (1−τ) · E[(m−Z)_+]` with the closed-form partial
/// expectations `E[(Z−m)_+] = φ(m) − m(1−Φ(m))` and
/// `E[(m−Z)_+] = φ(m) + mΦ(m)`. Solved by bisection on the monotone balance
/// function. This is the exact location of the conditional expectile up to the
/// scale factor σ(x).
fn standard_normal_expectile(tau: f64) -> f64 {
    let balance = |m: f64| -> f64 {
        let upper = phi(m) - m * (1.0 - big_phi(m)); // E[(Z−m)_+]
        let lower = phi(m) + m * big_phi(m); // E[(m−Z)_+]
        tau * upper - (1.0 - tau) * lower
    };
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    // balance is decreasing in m (more mass below ⇒ lower side grows).
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn build_data(n: usize, seed: u64) -> (gam::data::EncodedDataset, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).expect("normal");
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth_mean(t) + truth_scale(t) * z.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (data, x, y)
}

/// Fit `family = "expectile(τ)"` and predict the conditional τ-expectile on a
/// dense grid.
fn fit_expectile_predict(
    data: &gam::data::EncodedDataset,
    tau: f64,
    x_grid: &[f64],
) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some(format!("expectile({tau})")),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &cfg).map_err(|e| format!("fit error: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("expectile fit returned a non-standard model".to_string());
    };
    if fit.fit.beta.iter().any(|v| !v.is_finite()) {
        return Err("non-finite expectile beta".to_string());
    }
    let mut m = Array2::<f64>::zeros((x_grid.len(), 2));
    for (i, &t) in x_grid.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("predict design rebuild: {e}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if pred.iter().any(|v| !v.is_finite()) {
        return Err("non-finite expectile prediction".to_string());
    }
    Ok(pred)
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (s / a.len() as f64).sqrt()
}

#[test]
fn expectile_gam_recovers_closed_form_heteroscedastic_expectiles() {
    init_parallelism();
    let n = 1200;
    let (data, _x_train, _y_train) = build_data(n, 20260613);

    let x_grid: Vec<f64> = (0..200).map(|i| i as f64 / 199.0).collect();
    let taus = [0.1_f64, 0.5, 0.9];

    // Closed-form ground-truth expectile curve for each τ.
    let truth: Vec<Vec<f64>> = taus
        .iter()
        .map(|&tau| {
            let e = standard_normal_expectile(tau);
            x_grid
                .iter()
                .map(|&t| truth_mean(t) + truth_scale(t) * e)
                .collect::<Vec<f64>>()
        })
        .collect();

    let mut fitted: Vec<Vec<f64>> = Vec::with_capacity(taus.len());
    for &tau in &taus {
        let pred = fit_expectile_predict(&data, tau, &x_grid)
            .unwrap_or_else(|e| panic!("expectile(τ={tau}) fit/predict failed: {e}"));
        fitted.push(pred);
    }

    // 1. Truth recovery. The truth surface spans ≈ truth_mean range (≈ 3.7) plus
    //    the scale-driven spread; an RMSE bar of 0.30 on the response scale is a
    //    tight, un-weakened recovery bound for n=1200 with REML smoothing.
    for (i, &tau) in taus.iter().enumerate() {
        let err = rmse(&fitted[i], &truth[i]);
        assert!(
            err < 0.30,
            "expectile(τ={tau}) curve RMSE {err:.4} vs closed-form truth exceeds 0.30; \
             the fitted τ-expectile does not track f(x)+σ(x)·e_τ"
        );
    }

    // 2. Strict ordering and heteroscedastic fanning. The 0.1/0.5/0.9 curves
    //    must be strictly ordered everywhere, and the upper–lower gap must grow
    //    with x (σ(x)=0.35+0.9x increases), which a single mean curve cannot do.
    for k in 0..x_grid.len() {
        assert!(
            fitted[0][k] < fitted[1][k] && fitted[1][k] < fitted[2][k],
            "expectile curves out of order at x={:.3}: {:.3} / {:.3} / {:.3}",
            x_grid[k],
            fitted[0][k],
            fitted[1][k],
            fitted[2][k]
        );
    }
    let gap_lo = fitted[2][0] - fitted[0][0];
    let gap_hi = fitted[2][x_grid.len() - 1] - fitted[0][x_grid.len() - 1];
    assert!(
        gap_hi > 1.5 * gap_lo,
        "expectile fan did not widen with the heteroscedastic scale: gap(x≈0)={gap_lo:.3}, \
         gap(x≈1)={gap_hi:.3} (expected the upper–lower spread to grow with σ(x))"
    );

    // 3. In-sample calibration: at the fitted τ-expectile the asymmetric residual
    //    balance must equal τ. Recompute on the training grid prediction.
    let (data2, x_train, y_train) = build_data(n, 20260613);
    for &tau in &taus {
        let pred = fit_expectile_predict(&data2, tau, &x_train).expect("train-grid expectile");
        let mut upper = 0.0_f64; // Σ (y−ê)_+
        let mut lower = 0.0_f64; // Σ (ê−y)_+
        for (yi, ei) in y_train.iter().zip(pred.iter()) {
            let r = yi - ei;
            if r > 0.0 {
                upper += r;
            } else {
                lower += -r;
            }
        }
        // Defining equation: τ·upper = (1−τ)·lower ⇒ τ̂ = upper/(upper+lower)·...
        // The fitted asymmetry implied by the residual split:
        let implied = (tau * upper) / (tau * upper + (1.0 - tau) * lower);
        // implied should be ≈ 0.5 when the estimating equation is balanced.
        assert!(
            (implied - 0.5).abs() < 0.06,
            "expectile(τ={tau}) estimating equation unbalanced: τ·Σ(y−ê)_+ vs (1−τ)·Σ(ê−y)_+ \
             gives split {implied:.3} (want ≈ 0.5); the LAWS fixed point did not converge"
        );
    }
}

/// Re-encode held-in rows through the same schema inference the CSV loader uses,
/// keeping the train subset numerically identical to the loaded data.
fn lidar_train_dataset(range: &[f64], logratio: &[f64]) -> gam::data::EncodedDataset {
    let headers = vec!["range".to_string(), "logratio".to_string()];
    let records: Vec<StringRecord> = range
        .iter()
        .zip(logratio)
        .map(|(&r, &y)| StringRecord::from(vec![format!("{r:.17e}"), format!("{y:.17e}")]))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode lidar train subset")
}

/// REAL-DATA held-out arm: fit the conditional τ-expectile of `logratio ~ s(range)`
/// on the canonical, strongly heteroscedastic `lidar` benchmark, training on a
/// deterministic 3/4 split and evaluating on the held-out 1/4.
///
/// There is no mature off-the-shelf comparator that fits *expectile* GAMs on
/// arbitrary data (mgcv/gamlss/quantreg fit means, quantiles, or full location-
/// scale densities, not Newey–Powell expectiles), so — per the suite's
/// reference-as-truth policy — the objective bar is the DEFINING PREDICTIVE
/// PROPERTY of a conditional τ-expectile evaluated OUT OF SAMPLE:
///
///   1. Out-of-sample calibration. At the predicted τ-expectile `ê_τ(x)` on
///      points the fit never saw, the asymmetric residual balance must satisfy
///      the estimating equation `τ·Σ(y−ê)_+ ≈ (1−τ)·Σ(ê−y)_+`, i.e. the implied
///      split `τ·U/(τ·U+(1−τ)·L) ≈ 0.5`, for τ ∈ {0.1, 0.5, 0.9}. A fit that
///      merely interpolated the training residual asymmetry (overfit) would fail
///      this on held-out data; a correctly smoothed expectile generalizes.
///   2. Held-out ordering. The three predicted curves stay strictly ordered
///      `ê_0.1 < ê_0.5 < ê_0.9` on the held-out covariate — the asymmetric
///      structure is a real, generalizing property of the conditional law.
///   3. Central predictor. On held-out rows the median expectile (τ=0.5) is a
///      better point predictor of `y` (lower RMSE) than either extreme, because
///      it targets the conditional center while the extremes deliberately do not.
#[test]
fn expectile_gam_held_out_calibration_on_lidar() {
    init_parallelism();

    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range: Vec<f64> = ds.values.column(col["range"]).to_vec();
    let logratio: Vec<f64> = ds.values.column(col["logratio"]).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // Deterministic interleaved 3/4 train, 1/4 test split (no RNG).
    let mut tr_range = Vec::new();
    let mut tr_logratio = Vec::new();
    let mut te_range = Vec::new();
    let mut te_logratio = Vec::new();
    for i in 0..n {
        if i % 4 == 0 {
            te_range.push(range[i]);
            te_logratio.push(logratio[i]);
        } else {
            tr_range.push(range[i]);
            tr_logratio.push(logratio[i]);
        }
    }
    let n_test = te_range.len();
    assert!(n_test > 30, "need a meaningful held-out set, got {n_test}");

    let train_ds = lidar_train_dataset(&tr_range, &tr_logratio);
    let train_col = train_ds.column_map();
    let train_range_idx = train_col["range"];
    let train_ncols = train_ds.headers.len();

    let taus = [0.1_f64, 0.5, 0.9];
    let mut held_out: Vec<Vec<f64>> = Vec::with_capacity(taus.len());
    for &tau in &taus {
        let cfg = FitConfig {
            family: Some(format!("expectile({tau})")),
            ..FitConfig::default()
        };
        let result = fit_from_formula("logratio ~ s(range)", &train_ds, &cfg)
            .unwrap_or_else(|e| panic!("expectile(τ={tau}) lidar train fit failed: {e}"));
        let FitResult::Standard(fit) = result else {
            panic!("expectile(τ={tau}) returned a non-standard model");
        };
        assert!(
            fit.fit.beta.iter().all(|v| v.is_finite()),
            "non-finite expectile beta on lidar (τ={tau})"
        );

        // Predict the τ-expectile at the HELD-OUT range values (identity link =>
        // design*beta is the response-scale expectile). Points unseen in fitting.
        let mut test_grid = Array2::<f64>::zeros((n_test, train_ncols));
        for (i, &r) in te_range.iter().enumerate() {
            test_grid[[i, train_range_idx]] = r;
        }
        let design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .unwrap_or_else(|e| panic!("held-out design rebuild (τ={tau}): {e}"));
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "non-finite held-out expectile prediction (τ={tau})"
        );
        held_out.push(pred);
    }

    // 1. OUT-OF-SAMPLE CALIBRATION: the estimating equation balances on held-out
    //    data. The slack is a touch wider than the in-sample 0.06 because the
    //    n_test≈56 held-out asymmetric balance is noisier than the n=1200
    //    in-sample one; 0.10 is still a tight, un-weakened generalization bound
    //    (a mean fit would land at implied≈0.5 only for τ=0.5 and badly off for
    //    the extremes).
    for (i, &tau) in taus.iter().enumerate() {
        let mut upper = 0.0_f64; // Σ (y−ê)_+
        let mut lower = 0.0_f64; // Σ (ê−y)_+
        for (yi, ei) in te_logratio.iter().zip(held_out[i].iter()) {
            let r = yi - ei;
            if r > 0.0 {
                upper += r;
            } else {
                lower += -r;
            }
        }
        let implied = (tau * upper) / (tau * upper + (1.0 - tau) * lower);
        assert!(
            (implied - 0.5).abs() < 0.10,
            "held-out expectile(τ={tau}) on lidar is mis-calibrated: asymmetric-balance split \
             {implied:.3} (want ≈ 0.5); the fitted τ-expectile does not generalize the \
             conditional estimating equation to unseen points"
        );
    }

    // 2. HELD-OUT ORDERING: the asymmetric structure generalizes — curves stay
    //    strictly ordered on the held-out covariate.
    for k in 0..n_test {
        assert!(
            held_out[0][k] < held_out[1][k] && held_out[1][k] < held_out[2][k],
            "held-out expectile curves out of order at test point {k} (range={:.1}): \
             {:.3} / {:.3} / {:.3}",
            te_range[k],
            held_out[0][k],
            held_out[1][k],
            held_out[2][k]
        );
    }

    // 3. CENTRAL PREDICTOR: on held-out rows the median expectile (τ=0.5) is the
    //    best point predictor of y; the τ=0.1/0.9 expectiles deliberately track
    //    the lower/upper conditional bulk and so must have larger RMSE to y.
    let rmse_to_y = |pred: &[f64]| -> f64 {
        let s: f64 = pred
            .iter()
            .zip(te_logratio.iter())
            .map(|(p, y)| (p - y).powi(2))
            .sum();
        (s / n_test as f64).sqrt()
    };
    let rmse_lo = rmse_to_y(&held_out[0]);
    let rmse_mid = rmse_to_y(&held_out[1]);
    let rmse_hi = rmse_to_y(&held_out[2]);
    eprintln!(
        "lidar held-out expectile: n_train={} n_test={n_test} \
         RMSE_to_y[τ=0.1={rmse_lo:.4} τ=0.5={rmse_mid:.4} τ=0.9={rmse_hi:.4}]",
        tr_range.len()
    );
    assert!(
        rmse_mid < rmse_lo && rmse_mid < rmse_hi,
        "the held-out median expectile (τ=0.5) should be the best central point predictor of y: \
         RMSE τ=0.5={rmse_mid:.4} is not below both τ=0.1={rmse_lo:.4} and τ=0.9={rmse_hi:.4}"
    );
}
