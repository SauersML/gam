use gam::estimate::{CoefficientPriorMean, FitOptions, PenaltySpec, fit_gam_with_penalty_specs};
use gam::smooth::{
    CoefficientGroupPrior, CoefficientGroupSpec, CoefficientSelector, LinearTermSpec,
    TermCollectionSpec, build_term_collection_design, fit_term_collection_with_coefficient_groups,
};
use gam::types::{LikelihoodFamily, RhoPrior};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

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
fn overlapping_coefficient_groups_are_distinct_precision_coordinates() {
    let (x, _, _, _) = synthetic_two_score_data();
    let design = build_term_collection_design(x.view(), &two_linear_term_spec()).expect("design");
    let realized = design
        .realize_coefficient_groups(
            &[
                CoefficientGroupSpec {
                    name: "both_scores".to_string(),
                    selectors: vec![
                        CoefficientSelector::LinearTerm("score_a".to_string()),
                        CoefficientSelector::LinearTerm("score_b".to_string()),
                    ],
                    parent: None,
                    prior: Some(CoefficientGroupPrior::GammaPrecision {
                        shape: 2.0,
                        rate: 1.0,
                    }),
                    prior_mean: CoefficientPriorMean::Zero,
                },
                CoefficientGroupSpec {
                    name: "score_b_only".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("score_b".to_string())],
                    parent: None,
                    prior: Some(CoefficientGroupPrior::GammaPrecision {
                        shape: 4.0,
                        rate: 1.0,
                    }),
                    prior_mean: CoefficientPriorMean::Zero,
                },
            ],
            &RhoPrior::Flat,
        )
        .expect("overlapping groups");

    assert_eq!(realized.penalty_specs.len(), 2);
    assert_eq!(realized.nullspace_dims, vec![1, 2]);
    assert_eq!(
        realized.group_column_indices,
        vec![
            ("both_scores".to_string(), vec![1, 2]),
            ("score_b_only".to_string(), vec![2]),
        ]
    );
}

#[test]
fn cyclic_coefficient_group_hierarchy_is_rejected() {
    let (x, _, _, _) = synthetic_two_score_data();
    let design = build_term_collection_design(x.view(), &two_linear_term_spec()).expect("design");
    let err = design
        .realize_coefficient_groups(
            &[
                CoefficientGroupSpec {
                    name: "outer".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
                    parent: Some("inner".to_string()),
                    prior: None,
                    prior_mean: CoefficientPriorMean::Zero,
                },
                CoefficientGroupSpec {
                    name: "inner".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
                    parent: Some("outer".to_string()),
                    prior: None,
                    prior_mean: CoefficientPriorMean::Zero,
                },
            ],
            &RhoPrior::Flat,
        )
        .expect_err("cyclic hierarchy must fail");

    assert!(err.to_string().contains("cycle"));
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
        prior_mean: Default::default(),
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
        assert!(
            (a - b).abs() < 1e-8,
            "beta mismatch: grouped={a} manual={b}"
        );
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
            prior_mean: Default::default(),
        },
        CoefficientGroupSpec {
            name: "per_score_level".to_string(),
            selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
            parent: Some("publication_level".to_string()),
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 15.0,
                rate: 1.0,
            }),
            prior_mean: Default::default(),
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

#[test]
fn coefficient_group_constant_prior_mean_shrinks_toward_mean() {
    let n = 10;
    let mut x = Array2::<f64>::zeros((n, 2));
    let y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = if i % 2 == 0 { -0.2 } else { 0.2 };
        x[[i, 1]] = if i < n / 2 { -0.2 } else { 0.2 };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = two_linear_term_spec();
    let strong_precision = Some(CoefficientGroupPrior::GammaPrecision {
        shape: 250.0,
        rate: 1.0,
    });

    let toward_mean = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[CoefficientGroupSpec {
            name: "constant_mean".to_string(),
            selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
            parent: None,
            prior: strong_precision.clone(),
            prior_mean: CoefficientPriorMean::constant(array![2.0]),
        }],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("constant-mean fit");

    let toward_zero = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[CoefficientGroupSpec {
            name: "zero_mean".to_string(),
            selectors: vec![CoefficientSelector::LinearTerm("score_a".to_string())],
            parent: None,
            prior: strong_precision,
            prior_mean: CoefficientPriorMean::Zero,
        }],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("zero-mean fit");

    assert!(toward_mean.fit.beta[1] > 1.0);
    assert!(toward_mean.fit.beta[1].abs() > toward_zero.fit.beta[1].abs() + 0.75);
}

#[test]
fn coefficient_group_kernel_basis_prior_mean_recovers_known_amplitude() {
    let n = 48;
    let mut x = Array2::<f64>::zeros((n, 2));
    let alpha = 1.7;
    let kernel_values = array![1.0, -0.5];
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a = (i as f64 - 23.5) / 12.0;
        let b = if i % 3 == 0 { -1.0 } else { 0.5 };
        x[[i, 0]] = a;
        x[[i, 1]] = b;
        y[i] = alpha * (kernel_values[0] * a + kernel_values[1] * b);
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = two_linear_term_spec();
    let kernel_values_for_closure = kernel_values.clone();

    let fit = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[CoefficientGroupSpec {
            name: "kernel_mean".to_string(),
            selectors: vec![
                CoefficientSelector::LinearTerm("score_a".to_string()),
                CoefficientSelector::LinearTerm("score_b".to_string()),
            ],
            parent: None,
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 500.0,
                rate: 1.0,
            }),
            prior_mean: CoefficientPriorMean::kernel_basis(
                array![0.25, 0.75],
                alpha,
                Arc::new(move |_| kernel_values_for_closure.clone()),
            ),
        }],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("kernel-basis prior mean fit");

    let beta = array![fit.fit.beta[1], fit.fit.beta[2]];
    let alpha_hat = beta.dot(&kernel_values) / kernel_values.dot(&kernel_values);
    assert!((alpha_hat - alpha).abs() < 0.05);
}

#[test]
fn coefficient_group_zero_prior_mean_matches_default_bits() {
    let (x, y, weights, offset) = synthetic_two_score_data();
    let spec = two_linear_term_spec();
    let group = |prior_mean: CoefficientPriorMean| CoefficientGroupSpec {
        name: "zero_equivalence".to_string(),
        selectors: vec![
            CoefficientSelector::LinearTerm("score_a".to_string()),
            CoefficientSelector::LinearTerm("score_b".to_string()),
        ],
        parent: None,
        prior: Some(CoefficientGroupPrior::GammaPrecision {
            shape: 2.0,
            rate: 1.0,
        }),
        prior_mean,
    };

    let default_fit = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[group(Default::default())],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("default prior mean fit");
    let explicit_zero_fit = fit_term_collection_with_coefficient_groups(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[group(CoefficientPriorMean::Zero)],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(RhoPrior::Flat),
    )
    .expect("explicit zero prior mean fit");

    assert_eq!(
        default_fit.fit.lambdas.len(),
        explicit_zero_fit.fit.lambdas.len()
    );
    for (a, b) in default_fit
        .fit
        .lambdas
        .iter()
        .zip(explicit_zero_fit.fit.lambdas.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in default_fit
        .fit
        .beta
        .iter()
        .zip(explicit_zero_fit.fit.beta.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
