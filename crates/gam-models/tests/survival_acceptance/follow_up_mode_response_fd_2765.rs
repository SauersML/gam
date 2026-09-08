//! gam#2765 / gam#2767 end-to-end gate: on a follow-up-varying slope the
//! criterion's coefficient mode response `dβ̂/dψ` must match a finite difference
//! of the fit's own β̂.
//!
//! The unit gates in `psi_terms_fd_tests` difference `D_β H[δ]` and
//! `D²_β H[u,v]` against the family's own joint Hessian, which is where the
//! defect was: `add_pullback_primary_hessian` pulled the row Hessian back
//! through ONE slope channel, so on a varying slope every consumer of that
//! pullback differentiated a different model. This gate closes the same defect
//! from the other end, through a real fit, because `D_β H` is what builds the
//! Jeffreys curvature `H_Φ` and its second-order completion — hence the operator
//! the mode response is solved against. A wrong pullback therefore shows up as a
//! wrong `dβ̂/dψ`, and that is a quantity the outer runner already publishes
//! beside its own Ridders-certified finite difference.
//!
//! Uses 400 observations from the recovery fixture, a Weibull baseline, and
//! four temporal slope basis functions. Both Weibull axes must have nonzero
//! mode responses and agree with the fit's own finite-difference audit to
//! relative error below `1e-5`.
//!
//! This grades the mode response, not the total outer gradient. They are
//! separate contracts: this fixture isolates the coefficient response that the
//! follow-up margin changes, while the complete profiled-gradient calculus is
//! covered by its own outer-gradient gates.

#[test]
fn survival_marginal_slope_follow_up_mode_response_matches_fd_2765() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let (data, _, _) = super::follow_up_varying_slope_2765::build_dataset(400);
    let config = gam_models::fit_orchestration::FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        slope_time_k: Some(4),
        slope_time_degree: 2,
        time_num_internal_knots: 3,
        baseline_target: "weibull".to_string(),
        spatial_optimization: gam_terms::smooth::SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    gam_solve::estimate::enable_outer_gradient_fd_capture_for_psi(2);
    // This audit grades the mode response at the seed, before the outer search
    // finishes. Recovery and saved-model replay have separate acceptance gates.
    let fit_result =
        gam_models::fit_orchestration::fit_from_formula("Surv(time, event) ~ 1", &data, &config);
    let audit = gam_solve::estimate::take_outer_gradient_fd_capture().unwrap_or_else(|| {
        panic!(
            "no mode-response audit was captured: {:?}",
            fit_result.err()
        )
    });
    assert_eq!(
        audit.psi_dim, 2,
        "the Weibull chart must exercise both psi axes"
    );
    let atoms = audit
        .decomposition
        .atoms()
        .expect("mode-response decomposition");
    eprintln!(
        "[2765-MODE] relative_errors={:?} max_errors={:?} steps={:?} objective_fd_uncertainty={:?} active_face_drift={:?}",
        atoms.mode_response_relative_error,
        atoms.mode_response_max_abs_error,
        audit.psi_steps,
        audit.psi_fd_uncertainty,
        audit.curvature.as_ref().map(|value| &value.face_drift_max_abs),
    );
    for axis in 0..audit.psi_dim {
        let analytic_norm = atoms.analytic_mode_response_norm[axis];
        let measured_norm = atoms.finite_difference_mode_response_norm[axis];
        let relative_error = atoms.mode_response_relative_error[axis];
        assert!(
            analytic_norm.is_finite() && analytic_norm > 1e-6,
            "psi {axis}: a zero mode response does not exercise the varying slope"
        );
        assert!(
            measured_norm.is_finite() && measured_norm > 1e-6,
            "psi {axis}: the finite-difference mode response must be nonzero"
        );
        assert!(
            relative_error.is_finite() && relative_error < 1e-5,
            "psi {axis}: mode-response relative error {relative_error:e}, max absolute error {:e}, analytic norm {analytic_norm:e}, measured norm {measured_norm:e}",
            atoms.mode_response_max_abs_error[axis]
        );
        eprintln!(
            "[2765-MODE] psi={axis} analytic_norm={analytic_norm:e} measured_norm={measured_norm:e} relative_error={relative_error:e}"
        );
    }
}
