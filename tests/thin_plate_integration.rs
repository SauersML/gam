use gam::basis::create_thin_plate_spline_basis_with_knot_count;
use gam::{FitOptions, LikelihoodFamily, fit_gam, predict_gam};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

#[test]
fn thin_plate_fit_gam_gaussian_fast_integration() {
    // Deterministic 2D grid.
    let nx = 12usize;
    let ny = 10usize;
    let n = nx * ny;
    let mut data = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);

    let mut row = 0usize;
    for ix in 0..nx {
        for iy in 0..ny {
            let x1 = ix as f64 / (nx as f64 - 1.0);
            let x2 = iy as f64 / (ny as f64 - 1.0);
            data[[row, 0]] = x1;
            data[[row, 1]] = x2;
            // Smooth nonlinear surface.
            y[row] = (std::f64::consts::PI * x1).sin() + 0.5 * (x2 - 0.5).powi(2);
            row += 1;
        }
    }

    let (tps, _knots) =
        create_thin_plate_spline_basis_with_knot_count(data.view(), 24).expect("TPS basis");

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let s_list = tps.penalty_matrices();

    let fit = fit_gam(
        tps.basis.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![0, 0],
        },
    )
    .expect("fit_gam with TPS should succeed");

    assert_eq!(fit.lambdas.len(), 2);
    assert_eq!(fit.beta.len(), tps.basis.ncols());
    assert!(fit.edf_total.is_finite());

    let pred = predict_gam(
        tps.basis.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("predict_gam should succeed");

    let mse = (&pred.mean - &y).mapv(|v| v * v).mean().unwrap_or(f64::INFINITY);
    assert!(
        mse < 5e-2,
        "TPS integration fit is too inaccurate, mse={mse:.6e}"
    );
}

#[test]
fn thin_plate_fit_gam_gaussian_simulated_train_test() {
    let n_train = 900usize;
    let n_test = 300usize;
    let mut rng = StdRng::seed_from_u64(20260226);
    let noise = Normal::new(0.0, 0.10).expect("normal params must be valid");

    let mut x_train = Array2::<f64>::zeros((n_train, 2));
    let mut y_train = Array1::<f64>::zeros(n_train);
    let mut y_train_true = Array1::<f64>::zeros(n_train);

    // Simulate a smooth 2D surface with mixed radial + interaction structure.
    for i in 0..n_train {
        let x1 = rng.random_range(-1.0..1.0);
        let x2 = rng.random_range(-1.0..1.0);
        x_train[[i, 0]] = x1;
        x_train[[i, 1]] = x2;

        let r2 = (x1 - 0.25).powi(2) + (x2 + 0.15).powi(2);
        let f = 1.1 * (-r2 / (2.0 * 0.38 * 0.38)).exp()
            + 0.45 * (std::f64::consts::PI * x1).sin()
            - 0.30 * x2
            + 0.25 * x1 * x2;
        y_train_true[i] = f;
        y_train[i] = f + noise.sample(&mut rng);
    }

    // Build train basis and retain knots for test-time reconstruction.
    let (tps_train, knots) =
        create_thin_plate_spline_basis_with_knot_count(x_train.view(), 30).expect("TPS basis");
    let weights = Array1::ones(n_train);
    let offset = Array1::zeros(n_train);
    let s_list = tps_train.penalty_matrices();

    let fit = fit_gam(
        tps_train.basis.view(),
        y_train.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            max_iter: 60,
            tol: 1e-6,
            // First penalty has TPS polynomial null space (d+1 => 3 in 2D),
            // second ridge penalty has none.
            nullspace_dims: vec![3, 0],
        },
    )
    .expect("fit_gam with TPS should succeed");

    assert_eq!(fit.lambdas.len(), 2);
    assert_eq!(fit.beta.len(), tps_train.basis.ncols());
    assert!(fit.edf_total.is_finite());

    // Test set with the same data-generating process.
    let mut x_test = Array2::<f64>::zeros((n_test, 2));
    let mut y_test_true = Array1::<f64>::zeros(n_test);
    for i in 0..n_test {
        let x1 = rng.random_range(-1.0..1.0);
        let x2 = rng.random_range(-1.0..1.0);
        x_test[[i, 0]] = x1;
        x_test[[i, 1]] = x2;
        let r2 = (x1 - 0.25).powi(2) + (x2 + 0.15).powi(2);
        y_test_true[i] = 1.1 * (-r2 / (2.0 * 0.38 * 0.38)).exp()
            + 0.45 * (std::f64::consts::PI * x1).sin()
            - 0.30 * x2
            + 0.25 * x1 * x2;
    }
    let tps_test =
        gam::basis::create_thin_plate_spline_basis(x_test.view(), knots.view()).expect("TPS test basis");
    let pred = predict_gam(
        tps_test.basis.view(),
        fit.beta.view(),
        Array1::zeros(n_test).view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("predict_gam should succeed");

    let mse_test = (&pred.mean - &y_test_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    assert!(
        mse_test < 0.12,
        "TPS simulated integration test is too inaccurate: mse_test={mse_test:.6e}"
    );
}
