//! #2267 — a dense outer evaluation decomposes its state's exact observed information once.
//!
//! The dense criterion prices `½log|A|` off one pencil eigensystem of `A`. Its gradient used
//! to decompose the same `A` again for each consumer entered without it: the rank-charge
//! derivative's dispersion, the log-determinant channel geometry, and a second rank-charge
//! derivative inside those channels. Probe 1161362 read one 7692-dimensional state
//! decomposed four times before its 900 s cap, about 140 s each, and the complete value and
//! gradient would have decomposed it once more. The evaluation now owns the block and hands
//! it to every consumer.
//!
//! The count pin reads the production counter the `[SAE-EXACT-DENSE]` line prints. Its
//! mutant is the no-geometry rank-charge route, a `None` block at the assembler's rank-charge
//! call, which must fail it at 1 gradient decomposition. The identity pin compares the
//! received block against a decomposition at the evaluation's own cache, which is what the
//! gradient paid for before: every priced derivative must agree bit for bit.

use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::*;
use ndarray::Array1;

/// A frozen fixed state: no inner iterations, so the value materializes `A` at the
/// fixture's own state and descends no saddle.
const INNER_MAX_ITER: usize = 0;
const LEARNING_RATE: f64 = 0.4;
const RIDGE: f64 = 1.0e-6;
/// Inner iterations for the envelope pin: a positive budget runs the basin envelope, which the
/// frozen `INNER_MAX_ITER` bypasses.
const ENVELOPE_INNER_MAX_ITER: usize = 8;

fn bits(values: &Array1<f64>) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_dense_outer_evaluation_decomposes_its_state_once_2267() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        target,
        None,
        rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE,
        RIDGE,
    );
    let route_rho = objective.baseline_rho.clone();
    let entered = super::construction::exact_a_pencil_decompositions_on_this_thread();
    let evaluation = objective
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("#2267: the fixture prices a dense criterion at its own rho");
    let priced = super::construction::exact_a_pencil_decompositions_on_this_thread();
    let gradient = objective
        .analytic_gradient_for_outer_evaluation(&route_rho, &evaluation)
        .expect("#2267: the dense route differentiates its own evaluation");
    let differentiated = super::construction::exact_a_pencil_decompositions_on_this_thread();
    // The bits let a lane compare this value and gradient against a build before #2267.
    println!(
        "[#2267] decompositions: value {}, gradient {}; cost bits {:016x}; gradient bits {:?}",
        priced - entered,
        differentiated - priced,
        evaluation.cost.to_bits(),
        gradient
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>(),
    );
    assert!(
        gradient.iter().any(|value| *value != 0.0),
        "#2267: the fixture's gradient must be live, else a skipped derivative decomposes nothing"
    );
    assert_eq!(
        priced - entered,
        1,
        "#2267: the dense value decomposes its state once"
    );
    assert_eq!(
        differentiated - priced,
        0,
        "#2267: the gradient reads the block its value priced and decomposes nothing"
    );
}

#[test]
fn the_evaluations_block_prices_what_a_fresh_decomposition_prices_2267() {
    let (mut term, target, rho) = threshold_gate_tiny_fixture(false);
    let (_value, loss, cache, received) = term
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE,
            RIDGE,
            true,
        )
        .expect("#2267: the fixture prices a dense criterion at its own rho");
    let received = received.expect("#2267: the dense criterion hands out the block it priced");
    // What the gradient decomposed before #2267: `A` materialized again at this cache.
    let fresh = term
        .materialize_dense_exact_a_geometry(&rho, target.view(), &cache)
        .expect("#2267: a fresh decomposition at the evaluation's state");

    // Before #2267 the rank-charge dispersion took the admission route.
    let admission_rank_charge = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, None)
        .expect("#2267: the admission-routed rank-charge derivative");
    let received_rank_charge = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, Some(&received))
        .expect("#2267: the rank-charge derivative off the evaluation's block");
    assert!(
        received_rank_charge.direct_rho.iter().any(|value| *value != 0.0)
            || received_rank_charge.theta.t.iter().any(|value| *value != 0.0),
        "#2267: the rank-charge derivative must be live on this fixture, else the identity \
         compares zeros"
    );
    for (label, received_values, admission_values) in [
        (
            "direct rho",
            &received_rank_charge.direct_rho,
            &admission_rank_charge.direct_rho,
        ),
        (
            "theta t",
            &received_rank_charge.theta.t,
            &admission_rank_charge.theta.t,
        ),
        (
            "theta beta",
            &received_rank_charge.theta.beta,
            &admission_rank_charge.theta.beta,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(admission_values),
            "#2267: the rank charge's {label} moved with the dispersion's divergence route"
        );
    }

    let received_channels = term
        .dense_exact_a_logdet_channels(
            target.view(),
            &rho,
            &cache,
            &received,
            &received_rank_charge.theta,
        )
        .expect("#2267: the log-determinant channels off the evaluation's block");
    let fresh_channels = term
        .dense_exact_a_logdet_channels(
            target.view(),
            &rho,
            &cache,
            &fresh,
            &admission_rank_charge.theta,
        )
        .expect("#2267: the log-determinant channels off a fresh decomposition");
    assert!(
        received_channels.logdet_trace.iter().any(|value| *value != 0.0)
            && received_channels.theta_adjoint.t.iter().any(|value| *value != 0.0),
        "#2267: the log-determinant trace and Γ must be live on this fixture"
    );
    for (label, received_values, fresh_values) in [
        (
            "log-determinant trace",
            &received_channels.logdet_trace,
            &fresh_channels.logdet_trace,
        ),
        (
            "Γ t",
            &received_channels.theta_adjoint.t,
            &fresh_channels.theta_adjoint.t,
        ),
        (
            "Γ beta",
            &received_channels.theta_adjoint.beta,
            &fresh_channels.theta_adjoint.beta,
        ),
        (
            "stationarity adjoint t",
            &received_channels.stationarity_adjoint.t,
            &fresh_channels.stationarity_adjoint.t,
        ),
        (
            "stationarity adjoint beta",
            &received_channels.stationarity_adjoint.beta,
            &fresh_channels.stationarity_adjoint.beta,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(fresh_values),
            "#2267: the {label} read off the evaluation's block differs from a fresh decomposition"
        );
    }

    let lambda_smooth = rho
        .lambda_smooth_vec()
        .expect("#2267: the fixture's smoothing strengths are finite");
    let solver = term
        .outer_gradient_arrow_solver(&cache, &lambda_smooth)
        .expect("#2267: the evaluation's outer solver factors");
    let components = |geometry: &DenseExactAGeometry| {
        term.analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            &solver,
            None,
            None,
            Some(geometry),
        )
        .expect("#2267: the dense gradient components")
    };
    let from_received = components(&received);
    let from_fresh = components(&fresh);
    for (label, received_values, fresh_values) in [
        ("explicit", &from_received.explicit, &from_fresh.explicit),
        (
            "log-determinant trace",
            &from_received.logdet_trace,
            &from_fresh.logdet_trace,
        ),
        ("occam", &from_received.occam, &from_fresh.occam),
        (
            "implicit response",
            &from_received.third_order_correction,
            &from_fresh.third_order_correction,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(fresh_values),
            "#2267: the gradient's {label} read off the evaluation's block differs from a fresh \
             decomposition"
        );
    }
}

fn decompositions() -> u64 {
    super::construction::exact_a_pencil_decompositions_on_this_thread()
}

/// #2267 — a value probe hands the gradient lane the evaluation it priced.
///
/// Job 1244553 (the 635-row example, K=8) priced `ρ₁` in the value lane, then priced it again
/// in the gradient lane from the probe's own handoff, to the same digits (`½log|A|=1.221477e4
/// ½log|Φ|=1.217244e4`), at about 330 s a pricing. The handoff now carries that evaluation.
///
/// - The count pin runs the value probe and then the gradient at the same ρ, and reads the
///   production counter: the probe decomposes the state once and the gradient lane not at
///   all. Its mutant, a handoff that drops the evaluation, fails at 1 gradient decomposition.
/// - The identity pin prices the same installed state through the route the gradient lane
///   paid before. The reused evaluation must return that route's cost and gradient bit for bit.
#[test]
fn a_value_probe_hands_the_gradient_lane_its_priced_evaluation_2267() {
    use gam_solve::rho_optimizer::OuterObjective;
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let rho_flat = rho.flat_coordinates();
    // The installed-state audit prices the fixture's own state: no inner iterations and no
    // amortized warm start, so a fresh route prices exactly the state the probe priced.
    let audit = || {
        SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE,
            RIDGE,
        )
        .for_installed_state_audit()
    };
    let mut probed = audit();
    let entered = decompositions();
    let probe_cost = probed
        .eval_cost(&rho_flat)
        .expect("#2267: the value probe prices the fixture at its own rho");
    let valued = decompositions();
    let sample = probed
        .eval(&rho_flat)
        .expect("#2267: the gradient lane differentiates the handed-off state");
    let differentiated = decompositions();
    println!(
        "[#2267] probe then gradient: decompositions value {}, gradient {}; probe cost bits \
         {:016x}, gradient-lane cost bits {:016x}",
        valued - entered,
        differentiated - valued,
        probe_cost.to_bits(),
        sample.cost.to_bits(),
    );
    assert!(
        probe_cost.is_finite(),
        "#2267: the fixture must have a defined score, else no evaluation is handed off"
    );
    assert!(
        sample.gradient.iter().any(|value| *value != 0.0),
        "#2267: the fixture's gradient must be live, else a skipped derivative decomposes nothing"
    );
    assert_eq!(
        valued - entered,
        1,
        "#2267: the value probe decomposes its state once"
    );
    assert_eq!(
        differentiated - valued,
        0,
        "#2267: the gradient lane differentiates the evaluation its probe priced"
    );
    assert_eq!(
        sample.cost.to_bits(),
        probe_cost.to_bits(),
        "#2267: the gradient lane returns the value its probe ranked"
    );

    let mut fresh = audit();
    let route_rho = fresh.baseline_rho.clone();
    let evaluation = fresh
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("#2267: the route prices the same installed state");
    let gradient = fresh
        .analytic_gradient_for_outer_evaluation(&route_rho, &evaluation)
        .expect("#2267: the route differentiates its own evaluation");
    assert_eq!(
        sample.cost.to_bits(),
        evaluation.cost.to_bits(),
        "#2267: the handed-off evaluation's cost differs from a fresh pricing of its state"
    );
    assert_eq!(
        bits(&sample.gradient),
        bits(&gradient),
        "#2267: the handed-off evaluation's gradient differs from a fresh pricing of its state"
    );
}

/// #2267 — the basin envelope hands the gradient lane its argmin's priced evaluation.
///
/// With inner iterations the value lane prices the envelope: the discovery trajectory and
/// each saved member. The envelope keeps the evaluation of the least value while the members
/// are priced, and hands it off only while that basin still holds the argmin's slot. The
/// gradient at the same ρ then decomposes nothing and returns the envelope value.
#[test]
fn the_basin_envelope_hands_the_gradient_lane_its_argmins_evaluation_2267() {
    use gam_solve::rho_optimizer::OuterObjective;
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let rho_flat = rho.flat_coordinates();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        target,
        None,
        rho,
        ENVELOPE_INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE,
        RIDGE,
    );
    let entered = decompositions();
    let envelope_value = objective
        .eval_cost(&rho_flat)
        .expect("#2267: the envelope prices the fixture at its own rho");
    let valued = decompositions();
    let telemetry = objective.probe_telemetry();
    let sample = objective
        .eval(&rho_flat)
        .expect("#2267: the gradient lane differentiates the envelope argmin");
    let differentiated = decompositions();
    println!(
        "[#2267] envelope then gradient: decompositions value {}, gradient {}; envelope evals \
         {}, members {}; envelope bits {:016x}, gradient-lane cost bits {:016x}",
        valued - entered,
        differentiated - valued,
        telemetry.basin_envelope_evals,
        telemetry.basin_max_members,
        envelope_value.to_bits(),
        sample.cost.to_bits(),
    );
    assert!(
        envelope_value.is_finite(),
        "#2267: the fixture must have a defined envelope value, else nothing is handed off"
    );
    assert!(
        telemetry.basin_envelope_evals == 1 && valued - entered >= 2,
        "#2267: the value lane must run the envelope and price its discovery and its seed \
         member, else the envelope's hand-off is not exercised (envelope evals {}, value \
         decompositions {})",
        telemetry.basin_envelope_evals,
        valued - entered
    );
    assert_eq!(
        differentiated - valued,
        0,
        "#2267: the gradient lane differentiates the evaluation the envelope priced"
    );
    assert_eq!(
        sample.cost.to_bits(),
        envelope_value.to_bits(),
        "#2267: the gradient lane returns the envelope value"
    );
}
