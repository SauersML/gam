use gam::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, FitOptions, LikelihoodFamily,
    MaternNu, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    fit_term_collection, predict_gam,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

fn simulate_duchon_regression(n: usize, d: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(20260224);
    let mut x = Array2::<f64>::zeros((n, d));
    let noise = Normal::new(0.0, 0.12).expect("normal params must be valid");
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);

    // Fixed center for a smooth radial bump.
    let mut c = vec![0.0f64; d];
    for (j, cj) in c.iter_mut().enumerate() {
        *cj = -0.35 + 0.07 * (j as f64);
    }

    for i in 0..n {
        for j in 0..d {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }

        let mut dist2 = 0.0;
        for j in 0..d {
            let delta = x[[i, j]] - c[j];
            dist2 += delta * delta;
        }

        let linear = 0.8 * x[[i, 0]] - 0.55 * x[[i, 1]] + 0.35 * x[[i, 2]];
        let radial_bump = 1.4 * (-dist2 / (2.0 * 0.42 * 0.42)).exp();
        let smooth_1d = 0.45 * (std::f64::consts::PI * x[[i, 3]]).sin();
        let f = linear + radial_bump + smooth_1d;
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }

    (x, y, y_true)
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d() {
    let n = 900usize;
    let d = 10usize;
    let (x, y, y_true) = simulate_duchon_regression(n, d);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_10d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 36 },
                    length_scale: 0.9,
                    nu: MaternNu::FiveHalves,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    double_penalty: true,
                },
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fitted = fit_term_collection(
        x.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
        },
    )
    .expect("Duchon term-collection fit should succeed");

    // Double penalty -> two lambdas (kernel + ridge) for the single Duchon term.
    assert_eq!(fitted.fit.lambdas.len(), 2);
    assert!(fitted.fit.edf_total.is_finite());

    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on fitted Duchon design should succeed");
    assert!(pred.mean.iter().all(|v| v.is_finite()));

    let mse_model = (&pred.mean - &y_true).mapv(|v| v * v).mean().unwrap_or(f64::INFINITY);
    let y_mean = y_true.mean().unwrap_or(0.0);
    let mse_baseline = y_true
        .iter()
        .map(|&v| {
            let d = v - y_mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        mse_model < 0.45 * mse_baseline,
        "Duchon integration fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}"
    );
}
