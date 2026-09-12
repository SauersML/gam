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
//! wrong `dβ̂/dψ`, and that is a quantity the outer runner publishes to an armed
//! seed observer beside the selected mode itself.
//!
//! The observer differences β̂ here, in the test, because production code takes
//! no finite difference (SPEC rule 2, #2901). β̂ is differenced at the step a
//! Ridders ladder over the criterion accepts, so the step is one something
//! justified rather than one something guessed (#2461). A coordinate pinned
//! against a face of the seed box is differenced by the one-sided three-point
//! rule its room allows.
//!
//! Uses 400 observations from the recovery fixture, a Weibull baseline, and
//! four temporal slope basis functions. Both Weibull axes must have nonzero
//! mode responses and agree with the finite difference to relative error below
//! `1e-5`.
//!
//! This grades the mode response, not the total outer gradient. They are
//! separate contracts: this fixture isolates the coefficient response that the
//! follow-up margin changes, while the complete profiled-gradient calculus is
//! covered by its own outer-gradient gates.

use gam_linalg::numeric_derivative::{RiddersConfig, StencilErrorPowers, ridders_from_stencil};
use gam_solve::estimate::outer_eval_capture::{
    OuterSeedOrder, OuterSeedProbe, observe_next_outer_seed,
};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// The seed's analytic mode response against its finite difference, per ψ axis.
struct SeedModeResponseAudit {
    psi_dim: usize,
    axes: Vec<AxisModeResponse>,
}

/// One ψ axis's analytic and finite-difference mode response, both at the step
/// the criterion's Ridders ladder accepted.
struct AxisModeResponse {
    step: f64,
    criterion_fd_uncertainty: f64,
    analytic_norm: f64,
    measured_norm: f64,
    relative_error: f64,
    max_abs_error: f64,
}

/// The criterion value and the selected coefficient mode at the seed displaced
/// by `offset` along θ coordinate `j`.
fn displaced_mode(
    probe: &mut dyn OuterSeedProbe,
    j: usize,
    offset: f64,
) -> Result<(f64, Array1<f64>), String> {
    let mut theta = probe.layout().seed.clone();
    theta[j] += offset;
    let evaluation = probe
        .evaluate(&theta, OuterSeedOrder::Value)
        .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))?;
    let (beta, _) = evaluation.selected_mode.ok_or_else(|| {
        format!("theta[{j}] displaced by {offset:e} published no selected coefficient mode")
    })?;
    Ok((evaluation.cost, beta))
}

/// The step the ladder accepted, or `fallback` when no rung produced a usable
/// extrapolant.
fn accepted_step(ladder_step: f64, fallback: f64) -> f64 {
    if ladder_step.is_finite() && ladder_step > 0.0 {
        ladder_step
    } else {
        fallback
    }
}

fn compare_mode_response(
    analytic_beta_dot: &Array1<f64>,
    measured_beta_dot: &Array1<f64>,
    step: f64,
    criterion_fd_uncertainty: f64,
) -> Result<AxisModeResponse, String> {
    if measured_beta_dot.len() != analytic_beta_dot.len() {
        return Err(format!(
            "coefficient-response length mismatch: analytic={} finite_difference={}",
            analytic_beta_dot.len(),
            measured_beta_dot.len()
        ));
    }
    let difference = analytic_beta_dot - measured_beta_dot;
    let analytic_norm = analytic_beta_dot.dot(analytic_beta_dot).sqrt();
    let measured_norm = measured_beta_dot.dot(measured_beta_dot).sqrt();
    Ok(AxisModeResponse {
        step,
        criterion_fd_uncertainty,
        analytic_norm,
        measured_norm,
        relative_error: difference.dot(&difference).sqrt()
            / analytic_norm.max(measured_norm).max(1e-12),
        max_abs_error: difference
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs())),
    })
}

/// Difference the criterion and β̂ along ψ axis `psi_j` of the probe's seed.
fn difference_mode_response(
    probe: &mut dyn OuterSeedProbe,
    psi_j: usize,
    seed_cost: f64,
    seed_beta: &Array1<f64>,
    analytic_beta_dot: &Array1<f64>,
) -> Result<AxisModeResponse, String> {
    let layout = probe.layout().clone();
    let j = layout.rho_dim + psi_j;
    let theta_j = layout.seed[j];
    let left_room = (theta_j - layout.lower[j]).max(0.0);
    let right_room = (layout.upper[j] - theta_j).max(0.0);
    let nominal_step = f64::EPSILON.powf(0.25) * (1.0 + theta_j.abs());
    // The coarsest rung has to fit in the room the box leaves, and the ladder
    // spans `2^9 = 512` below it.
    let ladder = |room: f64| RiddersConfig {
        initial_step: (1.0e-2 * (1.0 + theta_j.abs())).min(0.5 * room),
        shrink: 2.0,
        rungs: 10,
    };
    let mut refusal: Option<String> = None;
    let mut cost = |offset: f64| -> f64 {
        if refusal.is_some() {
            return f64::NAN;
        }
        match displaced_mode(&mut *probe, j, offset) {
            Ok((value, _)) => value,
            Err(reason) => {
                refusal = Some(reason);
                f64::NAN
            }
        }
    };
    if left_room >= nominal_step && right_room >= nominal_step {
        let measured = ridders_from_stencil(
            |h| (cost(h) - cost(-h)) / (2.0 * h),
            ladder(left_room.min(right_room)),
            StencilErrorPowers::Even,
        );
        if let Some(reason) = refusal {
            return Err(reason);
        }
        let step = accepted_step(measured.step, nominal_step);
        let (_, plus) = displaced_mode(probe, j, step)?;
        let (_, minus) = displaced_mode(probe, j, -step)?;
        let measured_beta_dot = (&plus - &minus) / (2.0 * step);
        compare_mode_response(
            analytic_beta_dot,
            &measured_beta_dot,
            step,
            measured.uncertainty,
        )
    } else if right_room >= left_room && right_room > 0.0 {
        // Pinned against the LOWER face: only the forward three-point rule is
        // evaluable. Its error has every power from `h²` on, and its coarsest
        // rung has to fit `2h` inside the room.
        let measured = ridders_from_stencil(
            |h| (-3.0 * seed_cost + 4.0 * cost(h) - cost(2.0 * h)) / (2.0 * h),
            ladder(0.5 * right_room),
            StencilErrorPowers::Consecutive,
        );
        if let Some(reason) = refusal {
            return Err(reason);
        }
        let step = accepted_step(measured.step, nominal_step.min(0.5 * right_room));
        let (_, one) = displaced_mode(probe, j, step)?;
        let (_, two) = displaced_mode(probe, j, 2.0 * step)?;
        let measured_beta_dot =
            (4.0_f64 * &one - &two - 3.0_f64 * seed_beta) / (2.0 * step);
        compare_mode_response(
            analytic_beta_dot,
            &measured_beta_dot,
            step,
            measured.uncertainty,
        )
    } else if left_room > 0.0 {
        // Pinned against the UPPER face: the backward mirror of the rule above.
        let measured = ridders_from_stencil(
            |h| (3.0 * seed_cost - 4.0 * cost(-h) + cost(-2.0 * h)) / (2.0 * h),
            ladder(0.5 * left_room),
            StencilErrorPowers::Consecutive,
        );
        if let Some(reason) = refusal {
            return Err(reason);
        }
        let step = accepted_step(measured.step, nominal_step.min(0.5 * left_room));
        let (_, one) = displaced_mode(probe, j, -step)?;
        let (_, two) = displaced_mode(probe, j, -2.0 * step)?;
        let measured_beta_dot =
            (3.0_f64 * seed_beta - 4.0_f64 * &one + &two) / (2.0 * step);
        compare_mode_response(
            analytic_beta_dot,
            &measured_beta_dot,
            step,
            measured.uncertainty,
        )
    } else {
        Err(format!("psi {psi_j}: the seed box collapses theta[{j}]"))
    }
}

/// Grade the seed's analytic mode response on every ψ axis.
fn audit_mode_response(probe: &mut dyn OuterSeedProbe) -> Result<SeedModeResponseAudit, String> {
    let layout = probe.layout().clone();
    let seed = probe
        .evaluate(&layout.seed, OuterSeedOrder::ValueAndGradient)
        .map_err(|error| format!("analytic evaluation at the seed: {error}"))?;
    if !seed.cost.is_finite() {
        return Err(format!("the seed criterion is not finite: {}", seed.cost));
    }
    let (component_cost, _) = seed
        .criterion_components
        .ok_or("the seed evaluation published no scalar criterion components")?;
    if component_cost.to_bits() != seed.cost.to_bits() {
        return Err(format!(
            "the published criterion components belong to another evaluation: \
             objective={:.17e} components={component_cost:.17e}",
            seed.cost
        ));
    }
    let (seed_beta, response_cols) = seed
        .selected_mode
        .ok_or("the seed evaluation published no selected coefficient mode")?;
    let response_cols = response_cols
        .ok_or("the seed evaluation published no extended-coordinate mode responses")?;
    if response_cols.nrows() != seed_beta.len() || response_cols.ncols() != layout.psi_dim {
        return Err(format!(
            "mode-response layout mismatch: beta_dim={} response_shape={}x{} psi_dim={}",
            seed_beta.len(),
            response_cols.nrows(),
            response_cols.ncols(),
            layout.psi_dim
        ));
    }
    let mut axes = Vec::with_capacity(layout.psi_dim);
    for psi_j in 0..layout.psi_dim {
        let analytic_beta_dot = response_cols.column(psi_j).mapv(|value| -value);
        axes.push(difference_mode_response(
            probe,
            psi_j,
            seed.cost,
            &seed_beta,
            &analytic_beta_dot,
        )?);
    }
    Ok(SeedModeResponseAudit {
        psi_dim: layout.psi_dim,
        axes,
    })
}

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

    let captured: Rc<RefCell<Option<Result<SeedModeResponseAudit, String>>>> =
        Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    observe_next_outer_seed(
        2,
        Box::new(
            move |probe: &mut dyn OuterSeedProbe| -> Result<(), gam_solve::estimate::EstimationError> {
                *sink.borrow_mut() = Some(audit_mode_response(probe));
                Ok(())
            },
        ),
    );
    // This audit grades the mode response at the seed, before the outer search
    // finishes. Recovery and saved-model replay have separate acceptance gates.
    let fit_result =
        gam_models::fit_orchestration::fit_from_formula("Surv(time, event) ~ 1", &data, &config);
    let audit = captured
        .borrow_mut()
        .take()
        .unwrap_or_else(|| {
            panic!(
                "the outer runner lent no seed probe: {:?}",
                fit_result.err()
            )
        })
        .unwrap_or_else(|reason| panic!("the seed probe refused: {reason}"));
    assert_eq!(
        audit.psi_dim, 2,
        "the Weibull chart must exercise both psi axes"
    );
    for (axis, response) in audit.axes.iter().enumerate() {
        eprintln!(
            "[2765-MODE] psi={axis} step={:e} criterion_fd_uncertainty={:e} analytic_norm={:e} \
             measured_norm={:e} relative_error={:e} max_abs_error={:e}",
            response.step,
            response.criterion_fd_uncertainty,
            response.analytic_norm,
            response.measured_norm,
            response.relative_error,
            response.max_abs_error,
        );
        assert!(
            response.analytic_norm.is_finite() && response.analytic_norm > 1e-6,
            "psi {axis}: a zero mode response does not exercise the varying slope"
        );
        assert!(
            response.measured_norm.is_finite() && response.measured_norm > 1e-6,
            "psi {axis}: the finite-difference mode response must be nonzero"
        );
        assert!(
            response.relative_error.is_finite() && response.relative_error < 1e-5,
            "psi {axis}: mode-response relative error {:e}, max absolute error {:e}, \
             analytic norm {:e}, measured norm {:e}",
            response.relative_error,
            response.max_abs_error,
            response.analytic_norm,
            response.measured_norm,
        );
    }
}
