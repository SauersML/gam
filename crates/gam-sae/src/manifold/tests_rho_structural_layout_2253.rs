//! #2253 structural rho-layout regressions.
//!
//! A one-atom Softmax assignment is the singleton simplex: every row's gate is
//! exactly one, its entropy is exactly zero, and `log_lambda_sparse` is absent
//! from the mathematical objective. Hard TopK likewise has no sparsity penalty;
//! its fixed support is the sparsity mechanism. These are typed layout absences,
//! not optimizer-held coordinates.

use super::tests::{
    PlantedCircleAssignmentMode, planted_circle_embedded, planted_circle_seed_term,
};
use super::*;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, array};
use std::sync::Arc;

/// Deterministic K=1 periodic objective whose fitted dictionary carries real
/// non-constant signal. The four-row, one-output warm-start contract toy used
/// here previously is intentionally tiny and strongly regularized; its fitted
/// output lands at the column mean and correctly triggers the structural
/// fit-data-collapse ledger. It is therefore a poor active-basin derivative
/// witness; this fixture stays on a well-resolved noncollapsed branch.
fn planted_periodic_outer_objective_2253() -> SaeManifoldOuterObjective {
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    // `planted_circle_seed_term` installs the harmonic evaluator for basis
    // refresh. The analytic logdet-state adjoint also needs its second-jet view.
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 40, 1.0, 1.0e-6, 1.0e-6)
}

#[test]
fn fixed_assignment_strength_is_absent_from_flat_rho_layout_2253() {
    let softmax = SaeManifoldRho::new(-1.7, 0.4, vec![array![-0.2]])
        .for_assignment(AssignmentMode::softmax(0.8));
    assert_eq!(
        softmax.assignment_strength_layout(),
        AssignmentStrengthLayout::SoftmaxEntropy
    );
    assert_eq!(softmax.sparse_flat_index(), None);
    assert_eq!(softmax.smooth_flat_index(0), 0);
    assert_eq!(softmax.ard_flat_index(0, 0), 1);
    assert_eq!(softmax.to_flat(), array![0.4, -0.2]);

    // Reconstitution moves only the two mathematical coordinates and retains
    // the stored (inner-state) sparse value without emitting it into the outer
    // vector.
    let moved = array![0.7, -0.5];
    let restored = softmax.from_flat(moved.view());
    assert_eq!(restored.to_flat(), moved);
    assert_abs_diff_eq!(restored.log_lambda_sparse, -1.7, epsilon = 0.0);

    // TopK has no assignment-strength penalty at any K: the fixed support is
    // the sparsity constraint itself.
    let topk = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(AssignmentMode::top_k_support(1));
    assert_eq!(
        topk.assignment_strength_layout(),
        AssignmentStrengthLayout::FixedSupport
    );
    assert_eq!(topk.sparse_flat_index(), None);
    assert_eq!(topk.to_flat(), array![0.1, 0.1, 0.2, 0.3]);

    // Softmax regains the assignment-strength coordinate automatically when a
    // second atom makes entropy non-constant.
    let two_atom_softmax = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(AssignmentMode::softmax(0.8));
    assert_eq!(two_atom_softmax.sparse_flat_index(), Some(0));
    assert_eq!(two_atom_softmax.smooth_flat_index(0), 1);
}

#[test]
fn k1_softmax_active_rho_gradient_matches_directional_fd_2253() {
    let mut gradient_objective = planted_periodic_outer_objective_2253();
    let base = gradient_objective.baseline_rho.to_flat();
    assert_eq!(
        base.len(),
        2,
        "K=1 Softmax outer rho must contain only smoothness and ARD"
    );
    assert_eq!(gradient_objective.baseline_rho.sparse_flat_index(), None);

    // Match the real optimizer's Value -> ValueAndGradient handoff order.
    let base_cost = gradient_objective
        .eval_cost(&base)
        .expect("base value lane must converge");
    assert!(
        base_cost.is_finite(),
        "the active-rho derivative witness must start at feasible REML evidence: \
         base={base_cost:.17e}"
    );
    let evaluation = gradient_objective
        .eval(&base)
        .expect("base analytic-gradient lane must converge");
    assert!(
        !gradient_objective.term.frames_active(),
        "the focused identity witness must retain complete full-B theta coordinates"
    );
    assert!(
        evaluation.cost.is_finite(),
        "the analytic-gradient lane must price the same feasible REML basin: \
         cost={:.17e}",
        evaluation.cost
    );
    assert_eq!(evaluation.gradient.len(), 2);

    let rho_state = gradient_objective.baseline_rho.from_flat(base.view());
    let mut audit_term = gradient_objective.term.clone();
    let (_, audit_loss, audit_cache) = audit_term
        .reml_criterion_with_cache(
            gradient_objective.target.view(),
            &rho_state,
            gradient_objective.registry.as_ref(),
            0,
            gradient_objective.learning_rate,
            gradient_objective.ridge_ext_coord,
            gradient_objective.ridge_beta,
        )
        .expect("frozen accepted-state evidence audit must evaluate");
    let audit_solver = audit_term
        .outer_gradient_arrow_solver(&audit_cache, &rho_state.lambda_smooth_vec())
        .expect("frozen accepted-state outer solver");
    let audit_components = audit_term
        .analytic_outer_rho_gradient_components(
            gradient_objective.target.view(),
            &rho_state,
            &audit_loss,
            &audit_cache,
            &audit_solver,
        )
        .expect("frozen accepted-state gradient components");
    let plain_audit_solver = DeflatedArrowSolver::plain(&audit_cache);
    let plain_audit_components = audit_term
        .analytic_outer_rho_gradient_components(
            gradient_objective.target.view(),
            &rho_state,
            &audit_loss,
            &audit_cache,
            &plain_audit_solver,
        )
        .expect("plain-cache frozen gradient components");
    let deflated_row_directions: usize = audit_cache
        .deflated_row_directions
        .iter()
        .map(Vec::len)
        .sum();
    let mut kkt_term = audit_term.clone();
    let kkt_system = kkt_term
        .assemble_arrow_schur(
            gradient_objective.target.view(),
            &rho_state,
            gradient_objective.registry.as_ref(),
        )
        .expect("accepted-state KKT system");
    let kkt_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&kkt_system);
    let kkt_norm = kkt_norm_sq.sqrt();
    let quotient_kkt_norm = kkt_term.quotient_gradient_norm_from_system(
        &kkt_system,
        kkt_norm_sq,
        &rho_state.lambda_smooth_vec(),
    );
    let kkt_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * kkt_term.inner_iterate_scale();
    // Freeze one accepted base state for the fixed-theta derivative audit. Own
    // every closure input so the live objective remains mutably available to
    // the continuation probes below, and every directional/axis frozen FD is
    // evaluated at this same base state rather than whichever probe ran last.
    let frozen_anchor_term = gradient_objective.term.clone();
    let frozen_baseline_rho = gradient_objective.baseline_rho.clone();
    let frozen_target = gradient_objective.target.clone();
    let frozen_registry = gradient_objective.registry.clone();
    let frozen_learning_rate = gradient_objective.learning_rate;
    let frozen_ridge_ext_coord = gradient_objective.ridge_ext_coord;
    let frozen_ridge_beta = gradient_objective.ridge_beta;
    let criterion_parts_at = |rho_flat: &Array1<f64>, inner_max_iter: usize| {
        let mut term = frozen_anchor_term.clone();
        let rho = frozen_baseline_rho.from_flat(rho_flat.view());
        let (criterion, loss, cache) = term
            .reml_criterion_with_cache(
                frozen_target.view(),
                &rho,
                frozen_registry.as_ref(),
                inner_max_iter,
                frozen_learning_rate,
                frozen_ridge_ext_coord,
                frozen_ridge_beta,
            )
            .expect("frozen-state directional value probe");
        let extra_penalty = frozen_registry.as_ref().map_or(Ok(0.0), |registry| {
            term.reml_extra_penalty_value_total(registry)
                .map_err(|error| error.to_string())
        });
        let data_and_priors = loss.total() + extra_penalty.expect("frozen extra penalty");
        let half_logdet =
            0.5 * arrow_log_det_from_cache(&cache).expect("frozen authoritative log determinant");
        let mut htt_half = 0.0_f64;
        for row in 0..cache.undamped_factor_count() {
            let factor = cache.undamped_factor(row);
            for diagonal in 0..factor.nrows() {
                htt_half += factor[[diagonal, diagonal]].ln();
            }
        }
        let residual = term
            .reconstruction_residual(frozen_target.view(), &rho)
            .expect("criterion component audit residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))
            .expect("criterion component audit dispersion");
        let rank_dof = term
            .per_atom_realised_rank_dof(&rho, dispersion)
            .expect("criterion component audit rank dof");
        assert!(
            rank_dof.iter().all(|&dof| dof > 0.0),
            "the K=1 derivative witness must stay outside the rank-zero veto"
        );
        let n_eff = term.per_atom_effective_sample_size();
        let rank_charge = if term.soft_rank_charge() {
            term.per_atom_soft_learning_coefficient(&rho, dispersion)
                .expect("criterion component audit soft rank")
                .iter()
                .zip(n_eff.iter())
                .map(|(&coefficient, &n)| coefficient * n.max(1.0).ln())
                .sum()
        } else {
            rank_dof
                .iter()
                .zip(n_eff.iter())
                .map(|(&dof, &n)| 0.5 * dof * n.max(1.0).ln())
                .sum()
        };
        let negative_occam = -term.reml_occam_term(&rho).expect("frozen Occam value");
        let reconstructed =
            data_and_priors + half_logdet - htt_half + rank_charge + negative_occam;
        let roundoff = 64.0 * f64::EPSILON * (1.0 + criterion.abs().max(reconstructed.abs()));
        assert!(
            (criterion - reconstructed).abs() <= roundoff,
            "frozen criterion atoms must reconstruct exactly: criterion={criterion:.17e}, \
             reconstructed={reconstructed:.17e}, roundoff={roundoff:.3e}"
        );
        (
            (
                criterion,
                data_and_priors,
                half_logdet,
                htt_half,
                rank_charge,
                negative_occam,
            ),
            term,
        )
    };
    let frozen_parts_at = |rho_flat: &Array1<f64>| criterion_parts_at(rho_flat, 0).0;
    let frozen_cost_at = |rho_flat: &Array1<f64>| frozen_parts_at(rho_flat).0;

    let direction = array![0.6_f64, -0.8_f64];
    let analytic = evaluation.gradient.dot(&direction);
    let h = 1.0e-3_f64;
    let plus = &base + &(h * &direction);
    let minus = &base - &(h * &direction);
    // Probe both sides from the accepted base basin, exactly as the production
    // line search does. A fresh cold objective at each side differentiates its
    // seed-dependent basin-entry map rather than the branch whose analytic
    // derivative `evaluation.gradient` describes.
    let plus_cost = gradient_objective
        .eval_cost(&plus)
        .expect("+h directional value probe must converge");
    let plus_telemetry = gradient_objective.probe_telemetry();
    let minus_cost = gradient_objective
        .eval_cost(&minus)
        .expect("-h directional value probe must converge");
    let minus_telemetry = gradient_objective.probe_telemetry();
    let frozen_plus_cost = frozen_cost_at(&plus);
    let frozen_minus_cost = frozen_cost_at(&minus);
    assert!(
        plus_cost.is_finite(),
        "+h must evaluate feasible REML evidence: \
         cost={plus_cost:.17e}, telemetry={plus_telemetry:?}"
    );
    assert!(
        minus_cost.is_finite(),
        "-h must evaluate feasible REML evidence: \
         cost={minus_cost:.17e}, telemetry={minus_telemetry:?}"
    );
    assert_eq!(
        plus_telemetry.infeasible_criterion_evals, 0,
        "+h value probe returned infeasible evidence: {plus_telemetry:?}"
    );
    assert_eq!(
        minus_telemetry.infeasible_criterion_evals, 0,
        "-h value probe returned infeasible evidence: {minus_telemetry:?}"
    );
    assert_ne!(
        plus_cost.to_bits(),
        minus_cost.to_bits(),
        "the active-rho value lane is bit-flat across a nonzero directional step: \
         plus={plus_cost:.17e}, minus={minus_cost:.17e}, \
         plus_telemetry={plus_telemetry:?}, minus_telemetry={minus_telemetry:?}"
    );
    let finite_difference = (plus_cost - minus_cost) / (2.0 * h);
    let frozen_finite_difference = (frozen_plus_cost - frozen_minus_cost) / (2.0 * h);
    // Differentiate the accepted branch's actual inner state and compare it to
    // the implicit-function response used by the analytic log-det correction.
    // This is the decisive split between a bad IFT assembly and a bounded inner
    // driver that selects a different stationary root at rho +/- h.  Every
    // state probe starts from the same accepted center state; no cold-seed map
    // or prior directional probe is allowed to contaminate the response.
    let converged_state_at = |rho_flat: &Array1<f64>| {
        let mut term = frozen_anchor_term.clone();
        let rho = frozen_baseline_rho.from_flat(rho_flat.view());
        let (cost, _, _) = term
            .reml_criterion_with_cache(
                frozen_target.view(),
                &rho,
                frozen_registry.as_ref(),
                40,
                frozen_learning_rate,
                frozen_ridge_ext_coord,
                frozen_ridge_beta,
            )
            .expect("accepted-branch state-response probe must converge");
        let t = term.assignment.flatten_ext_coords();
        let beta = term
            .flatten_factored_border()
            .expect("accepted-branch factored border");
        (cost, t, beta)
    };
    let (state_plus_cost, state_plus_t, state_plus_beta) = converged_state_at(&plus);
    let (state_minus_cost, state_minus_t, state_minus_beta) = converged_state_at(&minus);
    let theta_fd = SaeArrowVector {
        t: (&state_plus_t - &state_minus_t) / (2.0 * h),
        beta: (&state_plus_beta - &state_minus_beta) / (2.0 * h),
    };
    let gamma = audit_term
        .logdet_theta_adjoint(&rho_state, &audit_cache, &audit_solver)
        .expect("accepted-state logdet theta adjoint");
    let mut directional_rhs = audit_term
        .outer_rho_gradient_ift_rhs(&rho_state, 0, &audit_cache)
        .expect("smoothness IFT rhs");
    directional_rhs.t *= direction[0];
    directional_rhs.beta *= direction[0];
    let ard_rhs = audit_term
        .outer_rho_gradient_ift_rhs(&rho_state, 1, &audit_cache)
        .expect("ARD IFT rhs");
    directional_rhs.t.scaled_add(direction[1], &ard_rhs.t);
    directional_rhs.beta.scaled_add(direction[1], &ard_rhs.beta);
    let solved_response = audit_term
        .solve_exact_stationarity(
            &rho_state,
            frozen_target.view(),
            &audit_cache,
            &audit_solver,
            &directional_rhs,
        )
        .expect("accepted-state exact stationarity response");
    let actual_implicit_logdet = 0.5 * (gamma.t.dot(&theta_fd.t) + gamma.beta.dot(&theta_fd.beta));
    let predicted_implicit_logdet =
        -0.5 * (gamma.t.dot(&solved_response.t) + gamma.beta.dot(&solved_response.beta));
    let response_error_norm = (&theta_fd.t + &solved_response.t)
        .iter()
        .chain((&theta_fd.beta + &solved_response.beta).iter())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let theta_fd_norm = theta_fd
        .t
        .iter()
        .chain(theta_fd.beta.iter())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let solved_response_norm = solved_response
        .t
        .iter()
        .chain(solved_response.beta.iter())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let mut coordinate_fd = Array1::<f64>::zeros(base.len());
    let mut frozen_coordinate_fd = Array1::<f64>::zeros(base.len());
    let mut frozen_data_and_priors_fd = Array1::<f64>::zeros(base.len());
    let mut frozen_half_logdet_fd = Array1::<f64>::zeros(base.len());
    let mut frozen_occam_fd = Array1::<f64>::zeros(base.len());
    for coordinate in 0..base.len() {
        let mut axis = Array1::<f64>::zeros(base.len());
        axis[coordinate] = 1.0;
        let axis_plus = &base + &(h * &axis);
        let axis_minus = &base - &(h * &axis);
        let axis_plus_cost = gradient_objective
            .eval_cost(&axis_plus)
            .expect("coordinate +h value probe must converge");
        let axis_minus_cost = gradient_objective
            .eval_cost(&axis_minus)
            .expect("coordinate -h value probe must converge");
        coordinate_fd[coordinate] = (axis_plus_cost - axis_minus_cost) / (2.0 * h);
        let frozen_axis_plus = frozen_parts_at(&axis_plus);
        let frozen_axis_minus = frozen_parts_at(&axis_minus);
        frozen_coordinate_fd[coordinate] = (frozen_axis_plus.0 - frozen_axis_minus.0) / (2.0 * h);
        frozen_data_and_priors_fd[coordinate] =
            (frozen_axis_plus.1 - frozen_axis_minus.1) / (2.0 * h);
        frozen_half_logdet_fd[coordinate] = (frozen_axis_plus.2 - frozen_axis_minus.2) / (2.0 * h);
        frozen_occam_fd[coordinate] = (frozen_axis_plus.3 - frozen_axis_minus.3) / (2.0 * h);
    }
    let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
    assert!(
        (analytic - finite_difference).abs() <= 5.0e-3 * scale,
        "K=1 Softmax active-coordinate derivative mismatch: analytic direction={analytic:.9e}, \
         central FD={finite_difference:.9e}, error={:.3e}, scale={scale:.3e}, \
         gradient={:?}, coordinate_fd={coordinate_fd:?}, \
         frozen_coordinate_fd={frozen_coordinate_fd:?}, \
         frozen_data_and_priors_fd={frozen_data_and_priors_fd:?}, \
         frozen_half_logdet_fd={frozen_half_logdet_fd:?}, \
         frozen_occam_fd={frozen_occam_fd:?}, \
         explicit={:?}, trace={:?}, occam={:?}, adjoint={:?}, \
         plain_trace={:?}, solver_gauges={}, deflated_row_directions={deflated_row_directions}, \
         frozen_fd={frozen_finite_difference:.9e}, \
         actual_implicit_logdet={actual_implicit_logdet:.9e}, \
         predicted_implicit_logdet={predicted_implicit_logdet:.9e}, \
         theta_fd_norm={theta_fd_norm:.9e}, solved_response_norm={solved_response_norm:.9e}, \
         response_error_norm={response_error_norm:.9e}, \
         state_plus_cost={state_plus_cost:.17e}, state_minus_cost={state_minus_cost:.17e}, \
         kkt={kkt_norm:.9e}, quotient_kkt={quotient_kkt_norm:.9e}, \
         kkt_tolerance={kkt_tolerance:.9e}, \
         base_cost={base_cost:.17e}, plus_cost={plus_cost:.17e}, \
         minus_cost={minus_cost:.17e}",
        (analytic - finite_difference).abs(),
        evaluation.gradient,
        audit_components.explicit,
        audit_components.logdet_trace,
        audit_components.occam,
        audit_components.third_order_correction,
        plain_audit_components.logdet_trace,
        audit_solver.gauge_basis.len(),
    );
}
