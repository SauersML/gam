use gam::construction::canonicalize_penalty_spec;
use gam::estimate::PenaltySpec;
use gam::estimate::FitOptions;
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
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

    let basis = create_thin_plate_spline_basis_with_knot_count(data.view(), 24).expect("TPS basis");
    let tps = basis.0;

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let s_list = vec![
        BlockwisePenalty::new(0..tps.basis.ncols(), tps.penalty_bending.clone()),
        BlockwisePenalty::new(0..tps.basis.ncols(), tps.penalty_ridge.clone()),
    ];

    let fit = fit_gam(
        tps.basis.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![0, 0],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
    )
    .expect("fit_gam with TPS should succeed");

    assert_eq!(fit.lambdas.len(), 2);
    assert_eq!(fit.beta.len(), tps.basis.ncols());
    assert!(fit.edf_total().is_some_and(f64::is_finite));

    let pred_mean = tps.basis.dot(&fit.beta) + &offset;

    let mse = (&pred_mean - &y)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    assert!(
        mse < 5e-2,
        "TPS integration fit is too inaccurate, mse={mse:.6e}"
    );
}

