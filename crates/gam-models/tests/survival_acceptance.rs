//! Public survival fitting acceptance at the crate that owns the estimator.
//! The mathematical inference registrations match the public facade's startup;
//! the CPU lane does not require a GPU dispatch registration.

#[path = "survival_acceptance/covariate_constant_slope_2930.rs"]
mod covariate_constant_slope_2930;
#[path = "survival_acceptance/declared_latent_law_2923.rs"]
mod declared_latent_law_2923;
#[path = "survival_acceptance/joint_latent_law_2929.rs"]
mod joint_latent_law_2929;
#[path = "survival_acceptance/finite_law_score_units_3477.rs"]
mod finite_law_score_units_3477;
#[path = "survival_acceptance/follow_up_mode_response_fd_2765.rs"]
mod follow_up_mode_response_fd_2765;
#[path = "survival_acceptance/follow_up_varying_slope_2765.rs"]
mod follow_up_varying_slope_2765;
#[path = "survival_acceptance/landmark_short_horizon_calibration_2336.rs"]
mod landmark_short_horizon_calibration_2336;
#[path = "survival_acceptance/latent_frailty_inner_solve_2714.rs"]
mod latent_frailty_inner_solve_2714;
#[path = "survival_acceptance/latent_loaded_vs_unloaded_chart_2714.rs"]
mod latent_loaded_vs_unloaded_chart_2714;
#[path = "survival_acceptance/location_scale_heteroscedastic_globalization_1569.rs"]
mod location_scale_heteroscedastic_globalization_1569;
#[path = "survival_acceptance/margslope_face_criterion_fd_2894.rs"]
mod margslope_face_criterion_fd_2894;
#[path = "survival_acceptance/moving_law_certificate_2926.rs"]
mod moving_law_certificate_2926;
#[path = "survival_acceptance/timewiggle_seed_3061.rs"]
mod timewiggle_seed_3061;

fn initialize_cpu_fitting() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        drop(
            gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
                gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
            )),
        );
        drop(gam_problem::rho_posterior::set_rho_posterior_escalator(
            Box::new(gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator),
        ));
        // Match the public fitting startup's stack allowance for survival jets.
        rayon::ThreadPoolBuilder::new()
            .stack_size(64 << 20)
            .build_global()
            .expect("initialize the survival acceptance worker pool");
    });
}
