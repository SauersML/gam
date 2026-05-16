use gam::basis::{PeriodicSpline1DOptions, fit_periodic_spline_1d};
use ndarray::{Array1, Array2, array};

const TWO_PI: f64 = std::f64::consts::TAU;

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn periodic_spline_interpolates_anisotropic_ellipse_in_multi_output_space() {
    let n = 33;
    let mut u = Array1::<f64>::zeros(n);
    let mut y = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 * TWO_PI / n as f64;
        u[i] = t;
        // Strong anisotropic stretching: not a unit circle.
        y[[i, 0]] = 4.0 * t.cos();
        y[[i, 1]] = 0.35 * t.sin();
    }

    let spline = fit_periodic_spline_1d(u.view(), y.view(), PeriodicSpline1DOptions::new(TWO_PI))
        .expect("periodic ellipse fit");
    let fitted = spline.evaluate(u.view()).expect("ellipse eval at knots");

    assert_eq!(spline.ambient_dim(), 2);
    assert_eq!(spline.num_sites(), n);
    assert!(max_abs(&fitted, &y) < 1e-10);

    let seam = array![0.0, TWO_PI, -TWO_PI, 11.0 * TWO_PI];
    let seam_y = spline.evaluate(seam.view()).expect("seam eval");
    for row in 1..seam_y.nrows() {
        for col in 0..seam_y.ncols() {
            assert!((seam_y[[row, col]] - seam_y[[0, col]]).abs() < 1e-10);
        }
    }

    let deriv = spline
        .evaluate_derivative(array![0.0, TWO_PI].view())
        .expect("seam derivative");
    for col in 0..deriv.ncols() {
        assert!((deriv[[0, col]] - deriv[[1, col]]).abs() < 1e-10);
    }
}

#[test]
fn periodic_spline_handles_skewed_oval_embedded_in_3d() {
    let n = 48;
    let mut u = Array1::<f64>::zeros(n);
    let mut y = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 * TWO_PI / n as f64;
        u[i] = t;
        // A distorted loop with mixed harmonics and a non-axis-aligned ambient embedding.
        let x = 2.0 * t.cos() + 0.22 * (3.0 * t).cos();
        let z = 0.8 * t.sin() - 0.15 * (2.0 * t).sin();
        y[[i, 0]] = x + 0.4 * z;
        y[[i, 1]] = -0.3 * x + 1.7 * z;
        y[[i, 2]] = 0.25 * (t + 0.3).sin() + 0.1 * (4.0 * t).cos();
    }

    let spline = fit_periodic_spline_1d(u.view(), y.view(), PeriodicSpline1DOptions::new(TWO_PI))
        .expect("periodic 3d oval fit");
    let fitted = spline.evaluate(u.view()).expect("oval eval at knots");
    assert_eq!(spline.ambient_dim(), 3);
    assert!(max_abs(&fitted, &y) < 1e-10);

    let dense_m = 97;
    let mut dense_u = Array1::<f64>::zeros(dense_m);
    for i in 0..dense_m {
        dense_u[i] = -1.2 * TWO_PI + i as f64 * 3.4 * TWO_PI / (dense_m - 1) as f64;
    }
    let dense_y = spline.evaluate(dense_u.view()).expect("dense wrapped eval");
    let shifted = dense_u.mapv(|v| v + 5.0 * TWO_PI);
    let shifted_y = spline.evaluate(shifted.view()).expect("shifted eval");
    assert!(max_abs(&dense_y, &shifted_y) < 1e-10);
}

#[test]
fn periodic_spline_accepts_unsorted_samples_and_duplicate_endpoint() {
    let u = array![TWO_PI, 0.5 * TWO_PI, 0.0, 1.5 * TWO_PI, 0.25 * TWO_PI];
    let mut y = Array2::<f64>::zeros((u.len(), 2));
    for i in 0..u.len() {
        let t = u[i];
        y[[i, 0]] = 1.5 * t.cos();
        y[[i, 1]] = 0.25 * t.sin();
    }

    let spline = fit_periodic_spline_1d(u.view(), y.view(), PeriodicSpline1DOptions::new(TWO_PI))
        .expect("fit with duplicate endpoint");
    assert_eq!(spline.num_sites(), 3);

    let at_zero = spline.evaluate(array![0.0].view()).expect("zero eval");
    let at_period = spline.evaluate(array![TWO_PI].view()).expect("period eval");
    assert!(max_abs(&at_zero, &at_period) < 1e-12);
    assert!((at_zero[[0, 0]] - 1.5).abs() < 1e-12);
    assert!(at_zero[[0, 1]].abs() < 1e-12);
}

#[test]
fn periodic_spline_rejects_scalar_only_or_degenerate_inputs() {
    let u = array![0.0, 1.0];
    let y = Array2::<f64>::zeros((2, 2));
    let err = fit_periodic_spline_1d(u.view(), y.view(), PeriodicSpline1DOptions::new(2.0))
        .expect_err("too few samples should fail");
    assert!(err.to_string().contains("at least three samples"));

    let u3 = array![0.0, 1.0, 2.0];
    let y_bad = Array2::<f64>::zeros((2, 2));
    let err = fit_periodic_spline_1d(u3.view(), y_bad.view(), PeriodicSpline1DOptions::new(3.0))
        .expect_err("row mismatch should fail");
    assert!(err.to_string().contains("does not match output row count"));

    let y3 = Array2::<f64>::zeros((3, 1));
    let err = fit_periodic_spline_1d(u3.view(), y3.view(), PeriodicSpline1DOptions::new(f64::NAN))
        .expect_err("bad period should fail");
    assert!(
        err.to_string()
            .contains("period must be finite and positive")
    );
}
