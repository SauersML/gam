use gam::estimate::FitOptions;
use gam::smooth::{
    CoefficientGroupSpec, CoefficientSelector, LinearCoefficientGeometry, LinearTermSpec,
    TermCollectionSpec, build_term_collection_designs_and_freeze_joint,
    build_term_collection_designs_joint,
    fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors,
};
use gam::types::{
    CoefficientGroupPrior, InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink,
};
use ndarray::{Array1, Array2};

fn opts() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 100,
        tol: 1e-9,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: RhoPrior::Flat,
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

#[test]
fn coefficient_groups_with_gamma_priors_add_distinct_penalty_coordinates() {
    let n = 48usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = (i as f64 - 12.0) / 5.0;
        let x1 = (i as f64 - 20.0) / 7.0;
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        y[i] = 2.0 * x0 - 0.5 * x1;
    }
    let spec = TermCollectionSpec {
        linear_terms: vec![
            LinearTermSpec {
                name: "x0".into(),
                feature_col: 0,
                feature_cols: vec![0],
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            },
            LinearTermSpec {
                name: "x1".into(),
                feature_col: 1,
                feature_cols: vec![1],
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            },
        ],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    let groups = vec![
        CoefficientGroupSpec {
            name: "g_x0".into(),
            selectors: vec![CoefficientSelector::LinearTerm("x0".into())],
            parent: None,
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 3.0,
                rate: 1.0,
            }),
            prior_mean: Default::default(),
        },
        CoefficientGroupSpec {
            name: "g_x1".into(),
            selectors: vec![CoefficientSelector::LinearTerm("x1".into())],
            parent: None,
            prior: Some(CoefficientGroupPrior::GammaPrecision {
                shape: 3.0,
                rate: 1.0,
            }),
            prior_mean: Default::default(),
        },
    ];

    let fit = fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors(
        data.view(), y.view(), Array1::ones(n).view(), Array1::zeros(n).view(), &spec, &groups,
        &[("x0".into(), 2.0, 1.0), ("x1".into(), 2.0, 1.0)],
        LikelihoodSpec::new(ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity)),
        &opts(),
    ).expect("Combining coefficient groups and block gamma priors should keep each group as a separate penalty coordinate in the fitted result.");

    assert_eq!(
        fit.fit.lambdas.len(),
        6,
        "Expected two base linear penalties, two double-penalty blocks, and two user coefficient-group penalties to remain distinct."
    );
}

#[test]
fn joint_design_build_freeze_keeps_spec_order_stable() {
    let data = Array2::<f64>::zeros((8, 1));
    let s1 = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "a".into(),
            feature_col: 0,
            feature_cols: vec![0],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    let s2 = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "b".into(),
            feature_col: 0,
            feature_cols: vec![0],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };

    let designs = build_term_collection_designs_joint(data.view(), &[s1.clone(), s2.clone()])
        .expect("Joint design materialization should preserve the caller ordering of input specs.");
    let (_frozen_designs, frozen_specs) =
        build_term_collection_designs_and_freeze_joint(data.view(), &[s1, s2]).expect(
            "Joint freeze should preserve index alignment between designs and resolved specs.",
        );

    assert_eq!(
        designs[0].linear_ranges[0].0, "a",
        "First design should correspond to first input spec."
    );
    assert_eq!(
        frozen_specs[1].linear_terms[0].name, "b",
        "Second frozen spec should correspond to second input spec."
    );
}
