use gam::estimate::FitOptions;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec, fit_term_collection_forspec,
    fit_term_collection_with_penalty_block_gamma_prior_callback,
    fit_term_collection_with_penalty_block_gamma_priors,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 160,
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: RhoPrior::Flat,
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

/// The penalty-block key for the fixture's single `LinearTermRidge` block.
///
/// This is the linear term's NAME. There is no term-KIND vocabulary: the label
/// candidates a block answers to are built from its own identity -- the global
/// penalty index (`0`, `penalty:0`), the term name (`x`, `x:0`), and the
/// penalty source (`LinearTermRidge`). These tests previously keyed `"linear"`,
/// which is none of those, so every one of them failed at
/// `realize_keyed_penalty_block_gamma_priors` before reaching its real
/// assertion. `every_documented_alias_selects_the_same_penalty_block` below
/// pins the whole alias set so a rename cannot reintroduce that silently.
const LINEAR_TERM_BLOCK: &str = "x";

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
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
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
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let base = fit_term_collection_forspec(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        likelihood.clone(),
        &opts,
    )
    .expect("base fit");
    let flat = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[(LINEAR_TERM_BLOCK.to_string(), 1.0, 0.0)],
        likelihood,
        &opts,
    )
    .expect("flat gamma fit");

    assert_eq!(base.fit.lambdas.as_slice(), flat.fit.lambdas.as_slice());
    assert_eq!(base.fit.beta.as_slice(), flat.fit.beta.as_slice());
    assert_eq!(base.fit.reml_score().expect("the fit reports a REML/LAML criterion").to_bits(), flat.fit.reml_score().expect("the fit reports a REML/LAML criterion").to_bits());
}

#[test]
fn informative_gamma_precision_prior_shrinks_by_map_update() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let fit = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[(LINEAR_TERM_BLOCK.to_string(), 100_001.0, 100.0)],
        likelihood,
        &opts,
    )
    .expect("informative gamma fit");

    let lambda = fit.fit.lambdas[0];
    let target_lambda = 1_000.0;
    assert!(
        ((lambda - target_lambda) / target_lambda).abs() < 0.05,
        "lambda={lambda}, expected near {target_lambda}"
    );

    // The ridge acts on the term's STANDARDIZED column, not on raw `x`, so the
    // shrinkage denominator is that column's cross-product and not `Sxx`.
    // `x` here is already mean-zero, and the standardization divides by the
    // POPULATION sd (`sd^2 = Sxx/n`), so `z = x/sd` has `sum z^2 = n` exactly,
    // whatever `Sxx` happens to be:
    //
    //     beta_raw = beta_unpenalized * n / (n + lambda)
    //
    // The superseded form used `Sxx = 5740` in place of `n = 41` and expected
    // 4.258 where the estimator gives 0.1969 -- a factor of 21.6. It had never
    // run: every test in this file died at the penalty-block label before
    // reaching an assertion, so the closed form was never checked against the
    // estimator it claims to describe.
    //
    // Feeding the OBSERVED lambda into the identity leaves the shrinkage law
    // itself as the only claim under test, which is what lets the tolerance be
    // 1e-4 instead of the 8% slop the old form needed. That also pins the
    // standardization convention: the sample-sd reading (`sum z^2 = n - 1`)
    // sits 2.3e-2 away and no longer passes.
    let n = 41.0_f64;
    let unpenalized_beta = 5.0_f64;
    let expected_beta = unpenalized_beta * n / (n + lambda);
    let observed_beta = fit.fit.beta[1];
    assert!(
        ((observed_beta - expected_beta) / expected_beta).abs() < 1e-4,
        "beta={observed_beta}, expected ridge shrinkage {expected_beta} at lambda={lambda}"
    );
}

#[test]
fn gamma_precision_prior_callback_is_invoked_once_per_penalty_block() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
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
        likelihood.clone(),
        &opts,
    )
    .expect("callback gamma fit");
    let keyed_fit = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[(LINEAR_TERM_BLOCK.to_string(), 13.0, 0.25)],
        likelihood,
        &opts,
    )
    .expect("keyed gamma fit");

    assert_eq!(
        seen,
        vec![(LINEAR_TERM_BLOCK.to_string(), 0, 1)],
        "the callback's `label` is the TERM NAME (`penalty_block_metadata` reads \
         `info.termname`), so it must agree with the key the keyed API accepts"
    );
    assert_eq!(
        callback_fit.fit.lambdas.as_slice(),
        keyed_fit.fit.lambdas.as_slice()
    );
    assert_eq!(
        callback_fit.fit.beta.as_slice(),
        keyed_fit.fit.beta.as_slice()
    );
}

/// Every alias a penalty block answers to must select the SAME block, and a
/// label outside that set must be refused loudly rather than silently ignored.
///
/// This is the guard whose absence let all three tests above ship keyed on
/// `"linear"` -- a label the API has never emitted. Nothing exercised the
/// vocabulary, so the mistake could only surface as an opaque runtime refusal
/// inside an unrelated assertion. Pinning the alias set through the public API
/// makes a rename of the term-name routing, the global-index spelling, or the
/// `LinearTermRidge` source fail here, at the vocabulary, with the offending
/// alias named.
#[test]
fn every_documented_alias_selects_the_same_penalty_block() {
    let (data, y, weights, offset, spec) = linear_fixture();
    let opts = fit_options();
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );

    // A prior informative enough that selecting the block visibly moves lambda,
    // so an alias that quietly matched NOTHING could not pass by coincidence.
    let shape = 100_001.0;
    let rate = 100.0;
    let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;
    for alias in ["x", "x:0", "penalty:0", "0", "LinearTermRidge"] {
        let fit = fit_term_collection_with_penalty_block_gamma_priors(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            &[(alias.to_string(), shape, rate)],
            likelihood.clone(),
            &opts,
        )
        .unwrap_or_else(|error| panic!("alias `{alias}` must select the block: {error}"));
        let observed = (fit.fit.lambdas.to_vec(), fit.fit.beta.to_vec());
        match &reference {
            None => {
                assert!(
                    (observed.0[0] - 1_000.0).abs() / 1_000.0 < 0.05,
                    "alias `{alias}` did not actually apply the prior: lambda={}",
                    observed.0[0]
                );
                reference = Some(observed);
            }
            Some(expected) => assert_eq!(
                &observed, expected,
                "alias `{alias}` selected a different block than `x`"
            ),
        }
    }

    let rejected = fit_term_collection_with_penalty_block_gamma_priors(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        &[("linear".to_string(), shape, rate)],
        likelihood,
        &opts,
    );
    let message = rejected
        .err()
        .map(|error| error.to_string())
        .expect("a label outside the block's alias set must be an error, never a silent no-op");
    assert!(
        message.contains("unknown Gamma precision hyperprior penalty block label(s): linear"),
        "the refusal must name the offending label: {message}"
    );
    assert!(
        message.contains("LinearTermRidge") && message.contains('x'),
        "the refusal must list the labels that WOULD have worked: {message}"
    );
}
