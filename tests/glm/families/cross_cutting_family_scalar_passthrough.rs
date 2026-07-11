use gam::estimate::{FitOptions, fit_gam};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

fn base_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
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
        persist_warm_start_disk: false,
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
            InverseLink::Standard(StandardLink::Log),
        ),
    );
    let tw_17 = fit_small(
        tweedie_y,
        LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: 1.7 },
            InverseLink::Standard(StandardLink::Log),
        ),
    );

    let tw_beta_delta = (&tw_15.beta - &tw_17.beta).mapv(f64::abs).sum();
    assert!(
        tw_beta_delta > 1e-6,
        "Tweedie beta should change with p; delta={tw_beta_delta}"
    );
    assert!(
        (tw_15.deviance - tw_17.deviance).abs() > 1e-8,
        "Tweedie deviance should change with p"
    );
    assert!(
        (tw_15.reml_score - tw_17.reml_score).abs() > 1e-8,
        "Tweedie REML score should change with p"
    );

    // Negative-Binomial `theta` is now ESTIMATED jointly with the mean (issue
    // #802), exactly like the Beta precision below — and unlike the Tweedie
    // variance power `p` above, which is a genuine *fixed structural* scalar. So
    // the construction-time `theta` is only a *seed* that must NOT leak into the
    // converged fit: the SAME data fit with two very different seed thetas must
    // yield an identical converged fit (coefficients, deviance, REML score) and
    // the same data-driven `theta_hat`. (Asserting "deviance changes with the
    // seed theta" — as this block previously did — is exactly the frozen-`theta`
    // behaviour #802 fixed: both seeds now converge to the same estimated theta.)
    let nb_y = array![0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 5.0, 7.0];
    let nb_spec_seed = |seed| {
        LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial {
                theta: seed,
                theta_fixed: false,
            },
            InverseLink::Standard(StandardLink::Log),
        )
    };
    let nb_seed2 = fit_small(nb_y.clone(), nb_spec_seed(2.0));
    let nb_seed8 = fit_small(nb_y.clone(), nb_spec_seed(8.0));

    let nb_beta_delta = (&nb_seed2.beta - &nb_seed8.beta).mapv(f64::abs).sum();
    assert!(
        nb_beta_delta < 1e-6,
        "NegBin fit must be independent of the seed theta (theta is estimated, #802); \
         coefficient delta={nb_beta_delta}"
    );
    assert!(
        (nb_seed2.deviance - nb_seed8.deviance).abs() < 1e-6,
        "NegBin deviance must be independent of the seed theta (theta is estimated); \
         {} vs {}",
        nb_seed2.deviance,
        nb_seed8.deviance
    );
    assert!(
        (nb_seed2.reml_score - nb_seed8.reml_score).abs() < 1e-6,
        "NegBin REML score must be independent of the seed theta (theta is estimated)"
    );
    let theta_seed2 = nb_seed2
        .likelihood_scale
        .negbin_theta()
        .expect("NegBin fit must carry an estimated theta");
    let theta_seed8 = nb_seed8
        .likelihood_scale
        .negbin_theta()
        .expect("NegBin fit must carry an estimated theta");
    assert!(
        (theta_seed2 - theta_seed8).abs() / theta_seed2.max(1e-12) < 1e-4,
        "estimated NegBin theta must be seed-independent; got {theta_seed2} vs {theta_seed8}"
    );

    // Beta is the analogous estimated case. Like the Negative-Binomial `theta`
    // above and unlike the Tweedie variance power `p` (a genuine *fixed
    // structural* scalar), the Beta precision `phi` is ESTIMATED jointly
    // with the mean (issues #567/#769/#770, all closed). The construction-time
    // `phi` is therefore only a *seed*: it must NOT leak into the converged fit
    // (a leaked seed was the #769 slope-bias bug), and the reported precision must
    // instead track the DATA's dispersion. So the correct passthrough contract for
    // Beta is the opposite of the fixed-scalar one: seed-independence of the fit,
    // plus a data-driven `phî`. (Asserting "deviance changes with the seed phi"
    // is what this block previously got wrong — both seeds converge to the same
    // estimated phi, hence the same deviance.)
    let beta_spec = || {
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 10.0 },
            InverseLink::Standard(StandardLink::Logit),
        )
    };
    let beta_spec_seed = |seed| {
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: seed },
            InverseLink::Standard(StandardLink::Logit),
        )
    };

    // (a) Seed-independence: the SAME data fit with two very different seed phis
    // must yield an identical converged fit — same coefficients, same deviance,
    // same REML score, and the same estimated precision.
    let beta_y = array![0.10, 0.15, 0.25, 0.35, 0.52, 0.63, 0.78, 0.88];
    let be_seed10 = fit_small(beta_y.clone(), beta_spec_seed(10.0));
    let be_seed30 = fit_small(beta_y.clone(), beta_spec_seed(30.0));

    let beta_delta = (&be_seed10.beta - &be_seed30.beta).mapv(f64::abs).sum();
    assert!(
        beta_delta < 1e-6,
        "Beta fit must be independent of the seed phi (phi is estimated, #769); \
         coefficient delta={beta_delta}"
    );
    assert!(
        (be_seed10.deviance - be_seed30.deviance).abs() < 1e-6,
        "Beta deviance must be independent of the seed phi (phi is estimated); \
         {} vs {}",
        be_seed10.deviance,
        be_seed30.deviance
    );
    assert!(
        (be_seed10.reml_score - be_seed30.reml_score).abs() < 1e-6,
        "Beta REML score must be independent of the seed phi (phi is estimated)"
    );
    let phi_seed10 = be_seed10
        .likelihood_scale
        .fixed_phi()
        .expect("Beta fit must carry an estimated precision");
    let phi_seed30 = be_seed30
        .likelihood_scale
        .fixed_phi()
        .expect("Beta fit must carry an estimated precision");
    assert!(
        (phi_seed10 - phi_seed30).abs() / phi_seed10.max(1e-12) < 1e-4,
        "estimated Beta phi must be seed-independent; got {phi_seed10} vs {phi_seed30}"
    );

    // (b) The estimated precision tracks the DATA's dispersion: responses tightly
    // concentrated about their (near-constant) mean carry far less Beta noise
    // (Var = mu(1-mu)/(1+phi)), so the estimated precision phî must be markedly
    // LARGER than for the widely-spread responses above. This is the data-driven
    // analogue of "the scale parameter feeds through to the fit".
    let beta_y_tight = array![0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55];
    let be_tight = fit_small(beta_y_tight, beta_spec());
    let phi_tight = be_tight
        .likelihood_scale
        .fixed_phi()
        .expect("Beta fit must carry an estimated precision");
    assert!(
        phi_tight > 2.0 * phi_seed10,
        "estimated Beta phi must track the data's dispersion: tight-data phî={phi_tight} \
         should far exceed spread-data phî={phi_seed10}"
    );
}
