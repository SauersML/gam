//! Quality gate for joint non-crossing multi-level expectile fits.
//!
//! # The capability
//!
//! `family = "expectile"` with several levels (`expectile_tau = [0.1, 0.5,
//! 0.9]`) returns ONE location-scale fit whose level curves are
//! `e_τ(x) = μ(x) + c_τ·σ(x)`, with `c_τ` strictly increasing in `τ` and
//! `σ(x) > 0` everywhere. Fitting each level on its own gives separately
//! penalized surfaces that are free to cross; the joint fit's curves are
//! ordered at every covariate value by construction, never by a post-hoc sort.
//!
//! # What is asserted
//!
//! 1. Under strong heteroscedasticity (σ shrinking to almost nothing at the
//!    right edge of the data) the separately fitted 0.1 and 0.9 expectile
//!    curves cross on a dense grid that extends past the data, while the joint
//!    fit's curves are strictly ordered at every grid point, and the joint
//!    curves track the closed-form truth `f(x) + σ(x)·e_τ` inside the data.
//! 2. In the homoscedastic case, where the location-scale model holds with a
//!    constant σ, the joint curves agree with the separate single-level fits.
//! 3. A one-level request is the single-level LAWS fit itself, and the saved
//!    joint model refuses estimator metadata that would break the ordering.

use csv::StringRecord;
use gam::inference::model::{FittedEstimator, FittedModel};
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::predict::input::build_predict_input_for_model;
use gam::predict::{
    FittedModelPredictExt, InferenceCovarianceMode, PosteriorMeanOptions, joint_expectile_curves,
};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const LEVELS: [f64; 3] = [0.1, 0.5, 0.9];

fn truth_mean(x: f64) -> f64 {
    1.0 + (3.0 * x).sin()
}

/// Strongly heteroscedastic scale: wide on the left, almost zero on the right.
fn heteroscedastic_scale(x: f64) -> f64 {
    0.02 + 1.2 * (1.0 - x).powi(2)
}

fn standard_normal_expectile(tau: f64) -> f64 {
    let phi = |z: f64| (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let balance = |m: f64| {
        let cdf = gam_math::probability::normal_cdf(m);
        tau * (phi(m) - m * (1.0 - cdf)) - (1.0 - tau) * (phi(m) + m * cdf)
    };
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
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

fn build_data(n: usize, seed: u64, scale: fn(f64) -> f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let y = truth_mean(x) + scale(x) * z.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit(data: &gam::data::EncodedDataset, config: &FitConfig) -> FittedModel {
    let payload = fit_formula_to_payload("y ~ s(x)".to_string(), data, config)
        .unwrap_or_else(|error| panic!("expectile fit failed: {error}"));
    let model = FittedModel::from_payload(payload);
    model
        .validate_for_persistence()
        .expect("a fitted expectile model validates for persistence");
    model
}

fn expectile_config(levels: &[f64]) -> FitConfig {
    FitConfig {
        family: Some("expectile".to_string()),
        expectile_tau: Some(levels.to_vec()),
        ..FitConfig::default()
    }
}

/// Posterior mean of the model's location surface at `grid`, plus the joint
/// expectile curves when the model is a joint fit.
fn predict(
    model: &FittedModel,
    data: &gam::data::EncodedDataset,
    grid: &[f64],
) -> (Array1<f64>, Option<Vec<(String, Array1<f64>)>>) {
    let frame = Array2::from_shape_fn((grid.len(), 2), |(i, j)| if j == 0 { grid[i] } else { 0.0 });
    let zero = Array1::<f64>::zeros(grid.len());
    let input = build_predict_input_for_model(
        model,
        frame.view(),
        &data.column_map(),
        model.training_headers.as_ref(),
        &zero,
        &zero,
        false,
    )
    .expect("predict input");
    let predictor = model.predictor().expect("a saved expectile model predicts");
    let mean = predictor
        .predict_posterior_mean(
            &input,
            model.unified().expect("a saved fit carries its unified fit"),
            &PosteriorMeanOptions {
                confidence_level: None,
                covariance_mode: InferenceCovarianceMode::Conditional,
                include_observation_interval: false,
                extrapolation_variance: None,
            },
        )
        .expect("posterior-mean prediction")
        .mean;
    let curves = joint_expectile_curves(model, &*predictor, &input, &mean).expect("joint curves");
    (mean, curves)
}

fn joint_curves(
    model: &FittedModel,
    data: &gam::data::EncodedDataset,
    grid: &[f64],
) -> Vec<Array1<f64>> {
    let (_, curves) = predict(model, data, grid);
    let curves = curves.expect("a multi-level request yields a joint expectile model");
    let names: Vec<&str> = curves.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["expectile_0.1", "expectile_0.5", "expectile_0.9"]);
    curves.into_iter().map(|(_, curve)| curve).collect()
}

fn separate_curves(data: &gam::data::EncodedDataset, grid: &[f64]) -> Vec<Array1<f64>> {
    LEVELS
        .iter()
        .map(|&tau| {
            let model = fit(data, &expectile_config(&[tau]));
            let (mean, curves) = predict(&model, data, grid);
            assert!(curves.is_none(), "a one-level request is a single LAWS fit");
            mean
        })
        .collect()
}

/// Dense grid over the data range `[0, 1]` extended by a quarter on each side.
fn dense_grid() -> Vec<f64> {
    (0..=600).map(|i| -0.25 + 1.5 * i as f64 / 600.0).collect()
}

#[test]
fn joint_expectile_curves_never_cross_where_separate_fits_do() {
    init_parallelism();
    let data = build_data(800, 20260919, heteroscedastic_scale);
    let grid = dense_grid();

    let separate = separate_curves(&data, &grid);
    let separate_crossings = (0..grid.len())
        .filter(|&k| separate[2][k] <= separate[0][k])
        .count();
    assert!(
        separate_crossings > 0,
        "fixture must be one where independently fitted 0.1/0.9 expectiles cross; they \
         stayed ordered at all {} grid points",
        grid.len()
    );

    let model = fit(&data, &expectile_config(&LEVELS));
    let joint = joint_curves(&model, &data, &grid);
    for k in 0..grid.len() {
        assert!(
            joint[0][k] < joint[1][k] && joint[1][k] < joint[2][k],
            "joint expectile curves out of order at x={:.4}: {:.5} / {:.5} / {:.5}",
            grid[k],
            joint[0][k],
            joint[1][k],
            joint[2][k]
        );
    }

    // Inside the data the joint curves recover the closed-form truth.
    for (curve, &tau) in joint.iter().zip(&LEVELS) {
        let e = standard_normal_expectile(tau);
        let inside: Vec<usize> = (0..grid.len())
            .filter(|&k| (0.0..=1.0).contains(&grid[k]))
            .collect();
        let mse = inside
            .iter()
            .map(|&k| {
                let x = grid[k];
                (curve[k] - truth_mean(x) - heteroscedastic_scale(x) * e).powi(2)
            })
            .sum::<f64>()
            / inside.len() as f64;
        assert!(
            mse.sqrt() < 0.12,
            "joint expectile τ={tau} RMSE {:.4} against f(x)+σ(x)·e_τ",
            mse.sqrt()
        );
    }
}

#[test]
fn joint_expectile_curves_match_separate_fits_when_homoscedastic() {
    init_parallelism();
    let data = build_data(800, 20260920, |_| 0.5);
    let grid: Vec<f64> = (0..=200).map(|i| i as f64 / 200.0).collect();

    let separate = separate_curves(&data, &grid);
    let model = fit(&data, &expectile_config(&LEVELS));
    let joint = joint_curves(&model, &data, &grid);
    for ((joint_curve, separate_curve), &tau) in joint.iter().zip(&separate).zip(&LEVELS) {
        let worst = joint_curve
            .iter()
            .zip(separate_curve)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // σ = 0.5; the two estimators target the same curve, so they must agree
        // to well inside the curves' own sampling error.
        assert!(
            worst < 0.08,
            "homoscedastic τ={tau}: joint and separate expectile curves differ by up to {worst:.4}"
        );
    }
}

#[test]
fn saved_joint_expectile_refuses_crossing_estimator_metadata() {
    init_parallelism();
    let data = build_data(300, 20260921, heteroscedastic_scale);
    let payload = fit_formula_to_payload("y ~ s(x)".to_string(), &data, &expectile_config(&LEVELS))
        .expect("joint expectile fit");
    let FittedEstimator::ExpectileLocationScale {
        levels,
        standardized_expectiles,
    } = payload.estimator.clone()
    else {
        panic!("a multi-level request saves a joint expectile estimator");
    };
    assert_eq!(levels, LEVELS);
    assert!(standardized_expectiles.windows(2).all(|pair| pair[0] < pair[1]));

    let mut swapped = standardized_expectiles.clone();
    swapped.swap(0, 2);
    let tampered = [
        FittedEstimator::ExpectileLocationScale {
            levels: levels.clone(),
            standardized_expectiles: swapped,
        },
        FittedEstimator::ExpectileLocationScale {
            levels: vec![0.9, 0.5, 0.1],
            standardized_expectiles: standardized_expectiles.clone(),
        },
        FittedEstimator::ExpectileLocationScale {
            levels: levels[..1].to_vec(),
            standardized_expectiles: standardized_expectiles[..1].to_vec(),
        },
        FittedEstimator::Likelihood,
    ];
    for estimator in tampered {
        let mut bad = payload.clone();
        bad.estimator = estimator.clone();
        assert!(
            FittedModel::from_payload(bad)
                .validate_for_persistence()
                .is_err(),
            "saved joint expectile model accepted estimator {estimator:?}"
        );
    }
}
