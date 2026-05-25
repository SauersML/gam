use gam::estimate::{fit_gam, FitOptions};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily};
use ndarray::{array, Array1, Array2};

fn base_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 50,
        tol: 1e-8,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn fit_small(y: Array1<f64>, spec: LikelihoodSpec) -> gam::estimate::UnifiedFitResult {
    let n = y.len();
    let mut x = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = i as f64 / (n as f64 - 1.0);
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let s_list: Vec<BlockwisePenalty> = vec![];
    fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        spec,
        &base_options(),
    )
    .expect("fit should succeed")
}

#[test]
fn cross_cutting_tweedie_negbin_beta_scalars_passthrough() {
    let tweedie_y = array![0.3, 0.7, 1.2, 1.8, 2.4, 3.1, 4.0, 5.2];
    let tw_15 = fit_small(
        tweedie_y.clone(),
        LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: 1.5 },
            InverseLink::Standard(LinkFunction::Log),
        ),
    );
    let tw_17 = fit_small(
        tweedie_y,
        LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: 1.7 },
            InverseLink::Standard(LinkFunction::Log),
        ),
    );

    let tw_beta_delta = (&tw_15.beta - &tw_17.beta).mapv(f64::abs).sum();
    assert!(tw_beta_delta > 1e-6, "Tweedie beta should change with p; delta={tw_beta_delta}");
    assert!((tw_15.deviance - tw_17.deviance).abs() > 1e-8, "Tweedie deviance should change with p");
    assert!((tw_15.reml_score - tw_17.reml_score).abs() > 1e-8, "Tweedie REML score should change with p");

    let nb_y = array![0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 5.0, 7.0];
    let nb_20 = fit_small(
        nb_y.clone(),
        LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial { theta: 2.0 },
            InverseLink::Standard(LinkFunction::Log),
        ),
    );
    let nb_80 = fit_small(
        nb_y,
        LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial { theta: 8.0 },
            InverseLink::Standard(LinkFunction::Log),
        ),
    );
    assert!((nb_20.deviance - nb_80.deviance).abs() > 1e-8, "NegBin deviance should change with theta");
    assert!((nb_20.reml_score - nb_80.reml_score).abs() > 1e-8, "NegBin REML score should change with theta");

    let beta_y = array![0.10, 0.15, 0.25, 0.35, 0.52, 0.63, 0.78, 0.88];
    let be_10 = fit_small(
        beta_y.clone(),
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 10.0 },
            InverseLink::Standard(LinkFunction::Logit),
        ),
    );
    let be_30 = fit_small(
        beta_y,
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 30.0 },
            InverseLink::Standard(LinkFunction::Logit),
        ),
    );
    assert!((be_10.deviance - be_30.deviance).abs() > 1e-8, "Beta deviance should change with phi");
    assert!((be_10.reml_score - be_30.reml_score).abs() > 1e-8, "Beta REML score should change with phi");
}
