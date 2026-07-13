use faer::Side;
use gam::faer_ndarray::FaerCholesky;
use gam::inference::functionals::{
    GaussianIdentityAverageDerivativeInput, average_derivative_gaussian_identity,
    penalty_times_beta,
};
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    BasisMetadata, OneDimensionalBoundary, PenaltySource, build_bspline_basis_1d,
    evaluate_bspline_derivative_scalar,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn bspline_design_and_derivative(x: &Array1<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 12,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let built = build_bspline_basis_1d(x.view(), &spec).expect("B-spline design");
    let design = built.design.to_dense();
    let penalty = built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("B-spline roughness penalty")
        .matrix
        .clone();
    let BasisMetadata::BSpline1D {
        knots,
        identifiability_transform,
        degree,
        ..
    } = built.metadata
    else {
        panic!("expected B-spline metadata");
    };
    assert!(
        identifiability_transform.is_none(),
        "test uses raw B-spline coefficients"
    );
    let degree = degree.expect("frozen B-spline degree");
    let mut derivative = Array2::<f64>::zeros(design.raw_dim());
    for (row_idx, &value) in x.iter().enumerate() {
        let mut row = derivative.row_mut(row_idx);
        let row_slice = row.as_slice_mut().expect("contiguous derivative row");
        evaluate_bspline_derivative_scalar(value, knots.view(), degree, row_slice)
            .expect("B-spline derivative row");
    }
    (design, derivative, penalty)
}

fn fit_penalized_gaussian(
    design: &Array2<f64>,
    y: &Array1<f64>,
    penalty: &Array2<f64>,
    lambda: f64,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let mut scaled_penalty = penalty.clone();
    scaled_penalty.mapv_inplace(|value| lambda * value);
    let mut hessian = design.t().dot(design);
    hessian += &scaled_penalty;
    let rhs = design.t().dot(y);
    let chol = hessian.cholesky(Side::Lower).expect("penalized Hessian");
    let beta = chol.solvevec(&rhs);
    let mu = design.dot(&beta);
    (beta, mu, scaled_penalty)
}

fn draw_sample(seed: u64) -> (Array1<f64>, Array1<f64>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let mut x_values: Vec<f64> = (0..220).map(|_| uniform.sample(&mut rng)).collect();
    x_values.sort_by(f64::total_cmp);
    let x = Array1::from_vec(x_values);
    let y = Array1::from_iter(
        x.iter()
            .map(|&value| (2.0 * std::f64::consts::PI * value).sin() + noise.sample(&mut rng)),
    );
    let truth = x
        .iter()
        .map(|&value| 2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * value).cos())
        .sum::<f64>()
        / x.len() as f64;
    (x, y, truth)
}

#[test]
fn average_derivative_onestep_corrects_oversmoothed_gaussian_spline() {
    gam::init_parallelism();
    let seeds = [941_u64, 942, 943, 944, 945];
    let mut plugin_violations = 0usize;
    for seed in seeds {
        let (x, y, truth) = draw_sample(seed);
        let (design, derivative_design, penalty) = bspline_design_and_derivative(&x);
        // λ chosen to put the fit in the *oversmoothed* regime where the plug-in
        // average derivative is biased toward zero and the one-step correction has
        // real work to do (see `plugin_violations >= 3` below). The B-spline
        // roughness penalty is normalized in the constrained frame (#1364/#1365/
        // #1366), so a literal λ here is far weaker than it was before that
        // rescale; λ≈1.5e3 reproduces the oversmoothing the original #1261
        // fixture obtained at the old, un-normalized λ=10.
        let (beta, mu, scaled_penalty) = fit_penalized_gaussian(&design, &y, &penalty, 1500.0);
        let penalty_beta = penalty_times_beta(scaled_penalty.view(), beta.view());
        let estimate =
            average_derivative_gaussian_identity(&GaussianIdentityAverageDerivativeInput {
                design: design.view(),
                derivative_design: derivative_design.view(),
                y: y.view(),
                mu: mu.view(),
                beta: beta.view(),
                penalty: scaled_penalty.view(),
                penalty_beta: penalty_beta.view(),
            })
            .expect("average derivative functional");

        let plugin_error = estimate.theta_plugin - truth;
        let onestep_error = estimate.theta_onestep - truth;
        if plugin_error.abs() > 0.15 {
            plugin_violations += 1;
        }
        assert!(
            onestep_error.abs() < plugin_error.abs(),
            "seed {seed}: one-step error {onestep_error:.4} should improve on plugin error {plugin_error:.4}"
        );
        assert!(
            onestep_error.abs() < 0.15,
            "seed {seed}: one-step average derivative error {onestep_error:.4}"
        );
        assert!(
            estimate.penalty_bias.signum() == (-plugin_error).signum(),
            "seed {seed}: penalty correction {:.4} should point opposite the observed plugin error {:.4}",
            estimate.penalty_bias,
            plugin_error
        );
        assert!(
            estimate.penalty_bias.abs() > 0.2 * plugin_error.abs()
                && estimate.penalty_bias.abs() < 1.2 * plugin_error.abs(),
            "seed {seed}: penalty correction {:.4} should be same order as plugin error {:.4}",
            estimate.penalty_bias,
            plugin_error
        );
        assert!(estimate.se > 0.0, "seed {seed}: SE must be positive");
        let z = onestep_error.abs() / estimate.se;
        assert!(z < 4.0, "seed {seed}: z-score {z:.3}");
        assert_eq!(estimate.n_effective, x.len());

        let direct_plugin = derivative_design
            .mean_axis(ndarray::Axis(0))
            .expect("non-empty derivative design")
            .dot(&beta);
        assert!(
            (direct_plugin - estimate.theta_plugin).abs() < 1e-10,
            "plugin estimate must equal a_theta^T beta"
        );
    }
    assert!(
        plugin_violations >= 3,
        "oversmoothed plugin should violate the 0.15 average-derivative bound on at least three seeds, got {plugin_violations}"
    );
}
