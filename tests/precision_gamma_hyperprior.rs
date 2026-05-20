use gam::estimate::FitOptions;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec, fit_term_collection_forspec,
    fit_term_collection_with_penalty_block_gamma_prior_callback,
    fit_term_collection_with_penalty_block_gamma_priors,
};
use gam::types::{LikelihoodFamily, RhoPrior};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 160,
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: RhoPrior::Flat,
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn linear_fixture() -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    TermCollectionSpec,
) {
    let n = 41usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x = i as f64 - 20.0;
        data[[i, 0]] = x;
        y[i] = 5.0 * x;
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    };
    (data, y, weights, offset, spec)
}

#[test]
fn flat_gamma_precision_prior_matches_uninformed_fit_bitwise() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let base = fit_term_collection_forspec(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &opts,
    )
    .expect("base fit");
    let flat = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[("linear".to_string(), 1.0, 0.0)],
        LikelihoodFamily::GaussianIdentity,
        &opts,
    )
    .expect("flat gamma fit");

    assert_eq!(base.fit.lambdas.as_slice(), flat.fit.lambdas.as_slice());
    assert_eq!(base.fit.beta.as_slice(), flat.fit.beta.as_slice());
    assert_eq!(base.fit.reml_score.to_bits(), flat.fit.reml_score.to_bits());
}

#[test]
fn informative_gamma_precision_prior_shrinks_by_map_update() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let fit = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[("linear".to_string(), 100_001.0, 100.0)],
        LikelihoodFamily::GaussianIdentity,
        &opts,
    )
    .expect("informative gamma fit");

    let lambda = fit.fit.lambdas[0];
    let target_lambda = 1_000.0;
    assert!(
        ((lambda - target_lambda) / target_lambda).abs() < 0.05,
        "lambda={lambda}, expected near {target_lambda}"
    );

    let sxx = (0..41)
        .map(|i| {
            let x = i as f64 - 20.0;
            x * x
        })
        .sum::<f64>();
    let expected_beta = 5.0 * sxx / (sxx + target_lambda);
    let observed_beta = fit.fit.beta[1];
    assert!(
        ((observed_beta - expected_beta) / expected_beta).abs() < 0.08,
        "beta={observed_beta}, expected shrinkage near {expected_beta}"
    );
}

#[test]
fn gamma_precision_prior_callback_is_invoked_once_per_penalty_block() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let mut seen = Vec::new();
    let callback_fit = fit_term_collection_with_penalty_block_gamma_prior_callback(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        |metadata| {
            seen.push((
                metadata.label.clone(),
                metadata.global_index,
                metadata.effective_rank,
            ));
            Some((13.0, 0.25))
        },
        LikelihoodFamily::GaussianIdentity,
        &opts,
    )
    .expect("callback gamma fit");
    let keyed_fit = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[("linear".to_string(), 13.0, 0.25)],
        LikelihoodFamily::GaussianIdentity,
        &opts,
    )
    .expect("keyed gamma fit");

    assert_eq!(seen, vec![("linear".to_string(), 0, 1)]);
    assert_eq!(
        callback_fit.fit.lambdas.as_slice(),
        keyed_fit.fit.lambdas.as_slice()
    );
    assert_eq!(
        callback_fit.fit.beta.as_slice(),
        keyed_fit.fit.beta.as_slice()
    );
}
