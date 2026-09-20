//! gam#3061 acceptance: a survival marginal-slope fit with `timewiggle(...)`
//! reaches its first inner solve and returns a fit.
//!
//! The time design's wiggle tail is a zero placeholder (the family evaluates the
//! warp dynamically), so the wiggle penalties act on the warp's Jacobian
//! `B(h₀(t_exit))`, not on those columns. The family used to re-seed every time
//! penalty against the placeholder design, whose mean Gram diagonal is exactly
//! zero, and refused every such fit with "the design's mean Gram diagonal is
//! 0e0" before any solve.

use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};

use super::follow_up_varying_slope_2765::build_dataset;

const N: usize = 1_200;

#[test]
fn survival_marginal_slope_with_a_timewiggle_is_seeded_and_fits_3061() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let (data, _times, _scores) = build_dataset(N);
    let cfg = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        time_num_internal_knots: 3,
        baseline_target: "weibull".to_string(),
        ..FitConfig::default()
    };

    let result = fit_from_formula("Surv(time, event) ~ 1 + timewiggle(internal_knots=2)", &data, &cfg)
        .expect("a survival marginal-slope fit with a timewiggle must not be refused at seed");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    let time_block = &fit.fit.blocks[0];
    assert!(
        time_block.beta.iter().all(|value| value.is_finite()),
        "the fitted time block must carry finite coefficients, got {:?}",
        time_block.beta
    );
    assert!(
        fit.fit.lambdas.iter().all(|lambda| lambda.is_finite() && *lambda > 0.0),
        "every smoothing parameter must be a finite positive value, got {:?}",
        fit.fit.lambdas
    );
}
