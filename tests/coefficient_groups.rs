use gam::estimate::{FitOptions, PenaltySpec, fit_gam_with_penalty_specs};
use gam::smooth::{
    CoefficientGroupPrior, CoefficientGroupSpec, CoefficientSelector, LinearTermSpec,
    TermCollectionSpec, build_term_collection_design, fit_term_collection_with_coefficient_groups,
};
use gam::types::{LikelihoodFamily, RhoPrior};
use ndarray::{Array1, Array2, array};

fn fit_options(rho_prior: RhoPrior) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior,
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn two_linear_term_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![
            LinearTermSpec {
                name: "score_a".to_string(),
                feature_col: 0,
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
            },
            LinearTermSpec {
                name: "score_b".to_string(),
                feature_col: 1,
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
            },
        ],
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

fn synthetic_two_score_data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = 40;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a = (i as f64 - 19.5) / 10.0;
        let b = if i % 2 == 0 { -0.75 } else { 0.75 };
        x[[i, 0]] = a;
        x[[i, 1]] = b;
        y[i] = 1.0 + 0.45 * a - 0.30 * b + 0.02 * ((i % 5) as f64 - 2.0);
    }
    (x, y, Array1::ones(n), Array1::zeros(n))
}

#[test]
fn coefficient_group_spanning_two_terms_matches_manual_merged_penalty() {
    let (x, y, weights, offset) = synthetic_two_score_data();
    let spec = two_linear_term_spec();
    let prior = CoefficientGroupPrior::GammaPrecision {
        shape: 2.0,
        rate: 1.0,
    };
    let groups = vec![CoefficientGroupSpec {
        name: "two_score_group".to_string(),
        selectors: vec![
            CoefficientSelector::LinearTerm("score_a".to_string()),
            CoefficientSelector::LinearTerm("score_b".to_string()),
        ],
        parent: None,
        prior: Some(prior),
    }];

    let grouped = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &groups,
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("grouped fit");

    let design = build_term_collection_design(x.view(), &spec).expect("manual design");
    let p = design.design.ncols();
    let mut merged = Array2::<f64>::zeros((p, p));
    merged[[1, 1]] = 1.0;
    merged[[2, 2]] = 1.0;
    let manual = fit_gam_with_penalty_specs(
        design.design,
        y.view(),
        weights.view(),
        offset.view(),
        vec![PenaltySpec::Dense(merged)],
        vec![p - 2],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::GammaPrecision {
            shape: 2.0,
            rate: 1.0,
        }),
    )
    .expect("manual merged fit");

    assert_eq!(grouped.fit.lambdas.len(), 1);
    assert!((grouped.fit.lambdas[0] - manual.lambdas[0]).abs() < 1e-8);
    for (a, b) in grouped.fit.beta.iter().zip(manual.beta.iter()) {
        assert!((a - b).abs() < 1e-8, "beta mismatch: grouped={a} manual={b}");
    }
}

#[test]
fn nested_groups_with_gamma_priors_apply_per_level_shrinkage() {
    let n = 64;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a = if i % 2 == 0 { -1.0 } else { 1.0 };
        let b = if (i / 2) % 2 == 0 { -1.0 } else { 1.0 };
        x[[i, 0]] = a;
        x[[i, 1]] = b;
        y[i] = 0.15 * a + 1.25 * b;
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = two_linear_term_spec();
    let groups = vec![
        CoefficientGroupSpec {
            name: "publication_level".to_string(),
            selectors: vec![
                CoefficientSelector::LinearTerm("score_a".to_string()),
                CoefficientSelector::LinearTerm("score_b".to_string()),
            ],
            parent: None,
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 3.0,
                rate: 1.0,
            }),
        },
        CoefficientGroupSpec {
            name: "per_score_level".to_string(),
            selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
            parent: Some("publication_level".to_string()),
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 15.0,
                rate: 1.0,
            }),
        },
    ];

    let fit = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &groups,
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("nested grouped fit");

    assert_eq!(fit.fit.lambdas.len(), 2);
    let parent_lambda = fit.fit.lambdas[0];
    let child_lambda = fit.fit.lambdas[1];
    assert!(
        child_lambda > parent_lambda,
        "child level should carry the stronger Gamma-prior precision: child={child_lambda} parent={parent_lambda}"
    );

    let unpenalized = array![0.0, 0.15, 1.25];
    assert!(fit.fit.beta[1].abs() < unpenalized[1].abs());
    assert!(fit.fit.beta[2].abs() > fit.fit.beta[1].abs() * 3.0);
}
