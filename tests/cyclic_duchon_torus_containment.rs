//! Regression: the fitted cyclic smooth must lie inside the data envelope.
//!
//! Data lives on a tilted 3D unit circle with Gaussian noise σ = 0.07. The
//! generating ground truth has range ~1.0 in each coordinate, so any fit that
//! stays sane should produce predictions within a generous 5σ ≈ 0.35 envelope
//! around the truth. We assert that condition.
//!
//! Several smooth families are run side-by-side. This locks in the quality
//! target that `duchon(ct, st)` and `duchon(ct, st, centers=80)` must stay in
//! the same sane envelope as thin-plate and Matérn smooths on the same cyclic
//! lift.

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

const TAU: f64 = std::f64::consts::TAU;

/// Generate noisy 3D circle data parameterized by theta ∈ [0, 2π).
/// Returns `(theta, clean[n,3], noisy[n,3])`.
fn make_noisy_circle(n: usize, seed: u64) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let utheta = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.07).expect("normal");
    let tilt = 30.0_f64.to_radians();
    let (st, ct) = tilt.sin_cos();

    let theta: Vec<f64> = (0..n).map(|_| utheta.sample(&mut rng)).collect();
    // Localized outward radial spike near θ = 2π/3 with width ~ 0.16 rad — a
    // sharp truth feature whose resolution stresses center-heavy smooths.
    let theta_spike = 2.0 * std::f64::consts::PI / 3.0;
    let spike_sigma = 0.16_f64;
    let spike_amp = 0.55_f64;

    let mut clean = Array2::<f64>::zeros((n, 3));
    let mut noisy = Array2::<f64>::zeros((n, 3));
    for (i, &t) in theta.iter().enumerate() {
        let dt = (t - theta_spike).sin().atan2((t - theta_spike).cos());
        let r = 1.0 + spike_amp * (-0.5 * (dt / spike_sigma).powi(2)).exp();
        let cx = r * t.cos();
        let cy = r * t.sin();
        // Rotate (cx, cy, 0) by `tilt` about x: y' = cy*cos − 0*sin, z' = cy*sin + 0*cos
        let x = cx;
        let y = cy * ct;
        let z = cy * st;
        clean[[i, 0]] = x;
        clean[[i, 1]] = y;
        clean[[i, 2]] = z;
        noisy[[i, 0]] = x + noise.sample(&mut rng);
        noisy[[i, 1]] = y + noise.sample(&mut rng);
        noisy[[i, 2]] = z + noise.sample(&mut rng);
    }
    (theta, clean, noisy)
}

/// Build a dataset with columns [ct, st, y].
fn build_dataset(theta: &[f64], y_col: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["ct", "st", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows = theta
        .iter()
        .zip(y_col.iter())
        .map(|(t, y)| {
            let ct = t.cos();
            let st = t.sin();
            StringRecord::from(vec![ct.to_string(), st.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Fit `y ~ <formula_body>` and return the absolute residual on the *training*
/// fitted values vs the clean ground truth for a single coordinate. We use the
/// training-time fitted values (X · β) rather than rebuilding a prediction
/// design at new points, because the bug we're catching shows up identically
/// at the training points: the LAML optimizer parks at a bad ρ and the
/// resulting β over- or under-fits visibly even at the training thetas.
/// Build a (n, 3) data array with columns [ct, st, y_placeholder] matching the
/// training layout so the frozen `resolvedspec` indexes into the right columns.
fn predict_data_matrix(theta_grid: &[f64]) -> Array2<f64> {
    let n = theta_grid.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, &t) in theta_grid.iter().enumerate() {
        m[[i, 0]] = t.cos();
        m[[i, 1]] = t.sin();
        m[[i, 2]] = 0.0; // y placeholder, not used by smooth term
    }
    m
}

fn max_residual_against_truth(
    theta_train: &[f64],
    y_noisy: &[f64],
    theta_test: &[f64],
    y_truth_test: &[f64],
    formula_body: &str,
) -> f64 {
    let data = build_dataset(theta_train, y_noisy);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ {}", formula_body);
    let result = fit_from_formula(&formula, &data, &cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    let beta = &fit.fit.beta;

    // Build prediction design at test points using the frozen spec.
    let new_data = predict_data_matrix(theta_test);
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild prediction design from frozen spec");
    assert_eq!(
        test_design.design.ncols(),
        beta.len(),
        "predict design width != beta length for `{}`",
        formula_body
    );
    let predicted = test_design.design.apply(beta);

    let mut max_abs = 0.0_f64;
    for (yhat, yt) in predicted.iter().zip(y_truth_test.iter()) {
        let r = (yhat - yt).abs();
        if r > max_abs {
            max_abs = r;
        }
    }
    max_abs
}

/// Generate the *clean* circle (with spike) at given thetas — for the truth
/// at prediction points.
fn clean_circle_at(theta: &[f64]) -> Array2<f64> {
    let n = theta.len();
    let tilt = 30.0_f64.to_radians();
    let (st, ct) = tilt.sin_cos();
    let theta_spike = 2.0 * std::f64::consts::PI / 3.0;
    let spike_sigma = 0.16_f64;
    let spike_amp = 0.55_f64;
    let mut clean = Array2::<f64>::zeros((n, 3));
    for (i, &t) in theta.iter().enumerate() {
        let dt = (t - theta_spike).sin().atan2((t - theta_spike).cos());
        let r = 1.0 + spike_amp * (-0.5 * (dt / spike_sigma).powi(2)).exp();
        let cx = r * t.cos();
        let cy = r * t.sin();
        clean[[i, 0]] = cx;
        clean[[i, 1]] = cy * ct;
        clean[[i, 2]] = cy * st;
    }
    clean
}

#[test]
fn cyclic_duchon_centers_80_fit_stays_within_envelope() {
    // 5σ envelope on the noise — a sane fit can never need more than this.
    const TOL: f64 = 0.35;
    init_parallelism();

    let (theta, _clean, noisy) = make_noisy_circle(260, 17);
    // Use the z coordinate (most diagnostic in our investigation)
    let y_noisy: Vec<f64> = noisy.column(2).to_vec();

    // Predict on a dense uniform grid spanning [0, 2π) — exposes wandering
    // between training points that in-sample residuals miss.
    let n_grid = 400usize;
    let theta_test: Vec<f64> = (0..n_grid)
        .map(|i| TAU * (i as f64) / (n_grid as f64))
        .collect();
    let clean_test = clean_circle_at(&theta_test);
    let y_truth_test: Vec<f64> = clean_test.column(2).to_vec();

    let cases: &[(&str, &str)] = &[
        ("thinplate", "thinplate(ct, st)"),
        ("matern", "matern(ct, st)"),
        ("duchon default", "duchon(ct, st)"),
        ("duchon 80", "duchon(ct, st, centers=80)"),
    ];

    let mut violations = Vec::new();
    for (label, body) in cases {
        let m = max_residual_against_truth(&theta, &y_noisy, &theta_test, &y_truth_test, body);
        eprintln!("[envelope] {label:20} max|pred - truth| on grid = {m:.4}");
        if m > TOL {
            violations.push(format!("{label} ({body}): {m:.4} > {TOL}"));
        }
    }
    assert!(
        violations.is_empty(),
        "fit wandered outside the {:.2}σ envelope:\n  - {}",
        TOL / 0.07,
        violations.join("\n  - "),
    );
}
