//! Regression test for issue #498.
//!
//! A tensor-product smooth with a periodic marginal
//! (`s(theta, h, periodic=[0], period=[2*pi, None], k=6)`) fits fine, but the
//! resolved/frozen spec produced at fit time could not be used for prediction:
//! the periodic Fourier margin was frozen with a `num_basis` of `q + 1` (read
//! from a `q + 1`-length placeholder knot vector) while the design realized at
//! fit time — and the frozen identifiability transform — used `q` columns. At
//! predict time the periodic axis rebuilt with one extra column per periodic
//! axis, so the tensor design no longer matched the frozen transform and the
//! prediction-design build aborted:
//!
//! ```text
//! Dimension mismatch: frozen tensor identifiability transform mismatch:
//! design has 42 columns but transform has 36 rows
//! ```
//!
//! These tests fit a noiseless cylinder surface, then predict from the
//! fitted/resolved spec. They assert the prediction-design builds, returns
//! finite values, and recovers the generating surface.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;

/// Noiseless cylinder surface: periodic in `theta`, linear in `h`.
fn cylinder_surface(theta: f64, h: f64) -> f64 {
    1.0 + 0.6 * theta.cos() - 0.25 * (2.0 * theta).sin() + 0.4 * h
}

/// Doubly-periodic torus surface: periodic in both axes.
fn torus_surface(a: f64, b: f64) -> f64 {
    0.5 + 0.7 * a.cos() + 0.3 * b.sin() - 0.2 * (a + b).cos()
}

fn encode_xyz(rows_xyz: &[(f64, f64, f64)]) -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = rows_xyz
        .iter()
        .map(|(a, b, y)| StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// Singly-periodic margin: `s(theta, h, periodic=[0], ...)`.
#[test]
fn tensor_periodic_margin_predict_recovers_cylinder_surface() {
    init_parallelism();

    // Grid over [0, 2pi) x [0, 1].
    let n_theta = 24usize;
    let n_h = 6usize;
    let mut training = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let theta = TWO_PI * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = (j as f64) / ((n_h - 1) as f64);
            training.push((theta, h, cylinder_surface(theta, h)));
        }
    }
    let data = encode_xyz(&training);

    let formula = "y ~ s(theta, h, periodic=[0], period=[2*pi, None], k=6)";
    let result = fit_from_formula(formula, &data, &gaussian_cfg()).expect("periodic tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // Predict at interior points (including theta values that exercise the
    // periodic wrap, e.g. near and across the 0/2pi seam).
    let test_thetas = [
        0.0,
        0.3,
        1.1,
        2.0,
        std::f64::consts::PI,
        4.5,
        6.0,
        TWO_PI - 1e-6,
    ];
    let test_hs = [0.1, 0.5, 0.9];
    let mut new_data = Array2::<f64>::zeros((test_thetas.len() * test_hs.len(), 2));
    let mut expected = Vec::with_capacity(new_data.nrows());
    let mut r = 0usize;
    for &theta in &test_thetas {
        for &h in &test_hs {
            new_data[[r, 0]] = theta;
            new_data[[r, 1]] = h;
            expected.push(cylinder_surface(theta, h));
            r += 1;
        }
    }

    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("fitted tensor periodic-margin prediction design failed to build");
    let preds = design.design.apply(&fit.fit.beta);

    let mut max_abs_err = 0.0f64;
    for (p, e) in preds.iter().zip(expected.iter()) {
        assert!(p.is_finite(), "prediction must be finite, got {p}");
        max_abs_err = max_abs_err.max((p - e).abs());
    }
    assert!(
        max_abs_err < 5e-2,
        "periodic tensor prediction should recover the noiseless cylinder surface; \
         max abs error = {max_abs_err}"
    );
}

/// Doubly-periodic margins: `s(a, b, periodic=[0,1], ...)` — both README
/// examples are affected by the off-by-one, so cover the two-periodic-axis
/// case from a different angle (every axis gets the +1 column bug).
#[test]
fn tensor_doubly_periodic_margin_predict_recovers_torus_surface() {
    init_parallelism();

    let n = 22usize;
    let mut training = Vec::with_capacity(n * n);
    for i in 0..n {
        let a = TWO_PI * (i as f64) / (n as f64);
        for j in 0..n {
            let b = TWO_PI * (j as f64) / (n as f64);
            training.push((a, b, torus_surface(a, b)));
        }
    }
    let data = encode_xyz(&training);

    let formula = "y ~ s(theta, h, periodic=[0,1], period=[2*pi, 2*pi], k=6)";
    let result =
        fit_from_formula(formula, &data, &gaussian_cfg()).expect("doubly-periodic tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    let test_pts = [
        (0.0, 0.0),
        (0.7, 5.9),
        (3.0, 1.0),
        (std::f64::consts::PI, std::f64::consts::PI),
        (TWO_PI - 1e-6, 0.2),
        (5.5, 4.5),
    ];
    let mut new_data = Array2::<f64>::zeros((test_pts.len(), 2));
    let mut expected = Vec::with_capacity(test_pts.len());
    for (r, &(a, b)) in test_pts.iter().enumerate() {
        new_data[[r, 0]] = a;
        new_data[[r, 1]] = b;
        expected.push(torus_surface(a, b));
    }

    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("fitted doubly-periodic prediction design failed to build");
    let preds = design.design.apply(&fit.fit.beta);

    let mut max_abs_err = 0.0f64;
    for (p, e) in preds.iter().zip(expected.iter()) {
        assert!(p.is_finite(), "prediction must be finite, got {p}");
        max_abs_err = max_abs_err.max((p - e).abs());
    }
    assert!(
        max_abs_err < 1e-1,
        "doubly-periodic tensor prediction should recover the noiseless torus surface; \
         max abs error = {max_abs_err}"
    );
}
