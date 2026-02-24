use gam::basis::create_thin_plate_spline_basis_with_knot_count;
use gam::{FitOptions, LikelihoodFamily, fit_gam, predict_gam};
use ndarray::{Array1, Array2};

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
