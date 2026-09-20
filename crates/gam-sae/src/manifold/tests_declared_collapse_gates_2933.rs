#![cfg(test)]

//! #2933 F05 — the analytic outer ρ-gradient must be the derivative of the value
//! the production objective reports, with every collapse-prevention gate decided by
//! production at every endpoint.
//!
//! The decoder-repulsion gate, the separation barrier's coactivation support and
//! the amplitude barrier's radius are functions `W(θ)` of the fitted state, and
//! every derivative of the criterion reads them as constants `w`. When each
//! evaluation re-derives them at its own root, the reported value is
//! `L(θ̂; W(θ̂)) + …`, whose ρ-derivative carries `L_w·W_θ·θ̂_ρ` through the implicit
//! Jacobian `L_θθ + L_θw·W_θ`. The analytic gradient contains neither, so it is not
//! the derivative of that value. An oracle that copies the centre's frozen gates onto
//! its endpoints differentiates the conditional objective and cannot see this; here
//! the objective runs its whole procedure at both endpoints.
//!
//! The control arm prices the same endpoints through the term-level criterion with
//! no declared gates, which re-derives the gates at each root. Its difference from
//! the analytic gradient is the endogenous-gate response at this fixture, and it
//! has to be material or the treatment arm would agree for the wrong reason.

use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

/// Two periodic atoms whose decoders write nearly the same output direction and
/// whose softmax routing splits every row between them, so the separation barrier
/// sees a co-firing, near-collinear pair and the repulsion gate engages.
fn co_firing_collinear_two_atom_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"));
    let coords0 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        (0.05 + 0.123 * row as f64).rem_euclid(1.0)
    });
    let coords1 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        (0.31 + 0.097 * row as f64).rem_euclid(1.0)
    });
    let atom = |name: &str, coords: &Array2<f64>, decoder: Array2<f64>| {
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic basis");
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .expect("basis, jet, decoder and Gram shapes agree")
        .with_basis_evaluator(evaluator.clone())
        .with_basis_second_jet(evaluator.clone())
    };
    let atom0 = atom(
        "periodic0",
        &coords0,
        array![[0.80, 0.10, 0.00], [0.50, 0.05, 0.00], [-0.40, -0.05, 0.00]],
    );
    let atom1 = atom(
        "periodic1",
        &coords1,
        array![[0.70, 0.25, 0.05], [0.45, 0.15, 0.00], [-0.35, -0.10, 0.02]],
    );
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        let phase = 0.9 * row as f64 + 1.3 * atom as f64;
        0.15 * phase.sin()
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .expect("logits, coordinate blocks and manifolds agree");
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment)
        .expect("atoms and assignment describe the same rows");
    let rho = SaeManifoldRho::new(0.4_f64.ln(), 0.5_f64.ln(), vec![array![0.0], array![0.0]]);
    let planted = term
        .try_fitted_for_rho(&rho)
        .expect("the fixture decodes at its own rho");
    let target = Array2::<f64>::from_shape_fn(planted.dim(), |(row, column)| {
        planted[[row, column]] + 0.05 * (1.3 * row as f64 + 0.7 * column as f64).sin()
    });
    (term, target, rho)
}

fn co_firing_collinear_two_atom_objective() -> SaeManifoldOuterObjective {
    let (term, target, rho) = co_firing_collinear_two_atom_fixture();
    SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6)
}

fn shifted(centre: &Array1<f64>, coordinate: usize, step: f64) -> Array1<f64> {
    let mut moved = centre.clone();
    moved[coordinate] += step;
    moved
}

#[test]
fn outer_gradient_is_the_derivative_of_the_value_production_reports_2933() {
    let mut objective = co_firing_collinear_two_atom_objective();
    let centre = objective.baseline_rho.flat_coordinates();
    let evaluation = objective
        .eval(&centre)
        .expect("the centre evaluation must return a (value, gradient) pair");
    assert!(
        evaluation.cost.is_finite(),
        "the fixture must price a finite criterion at its centre, got {}",
        evaluation.cost
    );
    let analytic = evaluation.gradient.clone();

    // Non-vacuity: at the priced root production's own gates are live.
    let repulsion_pairs = objective
        .term
        .decoder_repulsion_gate
        .as_ref()
        .map_or(0, Vec::len);
    let coactivation_pairs = objective
        .term
        .barrier_coactivation_gate
        .as_ref()
        .map_or(0, |gate| gate.pairs.len());
    let barrier = objective.term.separation_barrier_value(1.0);
    let repulsion = objective.term.decoder_repulsion_value(1.0);
    eprintln!(
        "[#2933 F05] centre cost={:.12e} analytic={analytic:?} repulsion_pairs={repulsion_pairs} \
         coactivation_pairs={coactivation_pairs} barrier={barrier:.6e} repulsion={repulsion:.6e}",
        evaluation.cost
    );
    assert!(
        coactivation_pairs > 0 && barrier > 0.0,
        "the fixture must price a live separation barrier on a co-firing pair \
         (pairs={coactivation_pairs}, barrier={barrier:e})"
    );

    // The accepted centre root with its declared gates released: the term-level
    // criterion then re-derives them at every root it prices.
    let mut accepted = objective.term.clone();
    accepted.decoder_repulsion_gate = None;
    accepted.barrier_coactivation_gate = None;
    accepted.amplitude_barrier_gate = None;
    accepted.streaming_gates_frozen = false;
    let layout = objective.baseline_rho.clone();
    let target = objective.target.clone();
    let (inner_max_iter, learning_rate, ridge_ext_coord, ridge_beta) = (
        objective.inner_max_iter,
        objective.learning_rate,
        objective.ridge_ext_coord,
        objective.ridge_beta,
    );
    let refreshed_value = |rho_flat: &Array1<f64>| {
        let rho = layout
            .from_flat(rho_flat.view())
            .expect("the endpoint uses the objective's layout");
        let mut endpoint = accepted.clone();
        endpoint
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .map(|(value, _, _)| value)
            .expect("the refreshed-gate endpoint must price a finite criterion")
    };

    const STEP: f64 = 2.0e-3;
    let mut worst_treatment = 0.0_f64;
    let mut worst_control = 0.0_f64;
    for coordinate in 0..centre.len() {
        let mut estimates = [0.0_f64; 2];
        for (index, step) in [STEP, 0.5 * STEP].into_iter().enumerate() {
            let plus = objective
                .eval_cost(&shifted(&centre, coordinate, step))
                .expect("the plus endpoint must evaluate");
            let minus = objective
                .eval_cost(&shifted(&centre, coordinate, -step))
                .expect("the minus endpoint must evaluate");
            assert!(
                plus.is_finite() && minus.is_finite(),
                "coordinate {coordinate}, step {step:e}: endpoints must be finite \
                 (plus={plus}, minus={minus})"
            );
            estimates[index] = (plus - minus) / (2.0 * step);
        }
        let richardson = (4.0 * estimates[1] - estimates[0]) / 3.0;
        let scale = 1.0 + analytic[coordinate].abs().max(richardson.abs());
        // The two step sizes disagree by the oracle's own truncation and roundoff;
        // that disagreement is charged to the same budget as the derivative gap.
        let oracle_error = (estimates[1] - estimates[0]).abs() / scale;
        let treatment = (analytic[coordinate] - richardson).abs() / scale + oracle_error;

        let control_fd = (refreshed_value(&shifted(&centre, coordinate, STEP))
            - refreshed_value(&shifted(&centre, coordinate, -STEP)))
            / (2.0 * STEP);
        let control = (analytic[coordinate] - control_fd).abs() / scale;
        eprintln!(
            "[#2933 F05] coordinate {coordinate}: analytic={:.10e} production_fd={estimates:?} \
             richardson={richardson:.10e} oracle_error={oracle_error:.3e} treatment_gap={treatment:.3e} \
             refreshed_gate_fd={control_fd:.10e} control_gap={control:.3e}",
            analytic[coordinate]
        );
        worst_treatment = worst_treatment.max(treatment);
        worst_control = worst_control.max(control);
    }
    eprintln!(
        "[#2933 F05] worst treatment gap={worst_treatment:.3e} worst control gap={worst_control:.3e}"
    );
    const TREATMENT_BUDGET: f64 = 1.0e-4;
    assert!(
        worst_control > 10.0 * TREATMENT_BUDGET,
        "control: the refreshed-gate value's derivative departs from the analytic gradient \
         by only {worst_control:.3e}; at this fixture the endogenous gates are not material \
         enough for the treatment arm to be free to disagree"
    );
    assert!(
        worst_treatment <= TREATMENT_BUDGET,
        "the analytic outer gradient is not the derivative of the value the objective \
         reports (worst relative gap {worst_treatment:.3e} > {TREATMENT_BUDGET:e}). With the \
         gates re-derived at each root the value is L(θ̂; W(θ̂)), whose derivative carries \
         L_w·W_θ·θ̂_ρ the frozen-gate gradient omits"
    );
}

/// A caller that froze a term's gates has declared them, and the objective holds
/// that set before any drive. A term with no frozen gates leaves the choice to the
/// first priced root.
#[test]
fn a_term_handed_in_with_frozen_gates_declares_them_2933() {
    let (mut term, target, rho) = co_firing_collinear_two_atom_fixture();
    term.refresh_decoder_repulsion_gate();
    term.refresh_barrier_coactivation_gate();
    term.refresh_amplitude_barrier_gate();
    term.streaming_gates_frozen = true;
    let frozen = term.collapse_prevention_gates();
    assert!(
        frozen
            .barrier_coactivation
            .as_ref()
            .is_some_and(|gate| !gate.pairs.is_empty()),
        "the co-firing fixture must carry a co-firing barrier pair: {frozen:?}"
    );
    let objective = SaeManifoldOuterObjective::new(
        term,
        target.clone(),
        None,
        rho.clone(),
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    assert_eq!(
        objective.collapse_prevention_gates.as_ref(),
        Some(&frozen),
        "an objective built on a term with frozen gates must declare exactly those gates"
    );

    let (unfrozen, target, rho) = co_firing_collinear_two_atom_fixture();
    let objective =
        SaeManifoldOuterObjective::new(unfrozen, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    assert!(
        objective.collapse_prevention_gates.is_none(),
        "with no frozen gates the objective must leave the choice to its first priced root"
    );
}

/// One objective prices every ρ under one gate set. The first finite root chooses
/// it. A committed analytic sample at another ρ reads it, although that root's own
/// gates differ. A reset keeps it. A reactive scalar waypoint installs another
/// objective and clears it, and a rollback restores it.
#[test]
fn one_objective_holds_one_gate_set_across_rho_reset_and_waypoint_rollback_2933() {
    let mut objective = co_firing_collinear_two_atom_objective();
    let centre = objective.baseline_rho.flat_coordinates();
    let first = objective
        .eval(&centre)
        .expect("the centre evaluation must return a (value, gradient) pair");
    assert!(first.cost.is_finite(), "the centre must price finitely: {}", first.cost);
    let declared = objective
        .collapse_prevention_gates
        .clone()
        .expect("the first finite priced root must declare the objective's gates");
    assert_eq!(objective.term.collapse_prevention_gates(), declared);
    assert!(
        objective.term.streaming_gates_frozen,
        "the driven term must hold the declared gates, not re-derive them"
    );

    let moved = shifted(&centre, 0, 0.3);
    let second = objective
        .eval(&moved)
        .expect("the moved evaluation must return a (value, gradient) pair");
    assert!(second.cost.is_finite(), "the moved rho must price finitely: {}", second.cost);
    assert_eq!(
        objective.collapse_prevention_gates.as_ref(),
        Some(&declared),
        "a second rho must not re-choose the objective's gates"
    );
    assert_eq!(
        objective.term.collapse_prevention_gates(),
        declared,
        "the analytic sample at the moved rho must be priced under the declared gates"
    );
    // Negative control: the moved root's own gates differ from the declared set, so
    // the equality above distinguishes holding the gates from re-deriving them.
    let mut rederived = objective.term.clone();
    rederived.refresh_decoder_repulsion_gate();
    rederived.refresh_barrier_coactivation_gate();
    rederived.refresh_amplitude_barrier_gate();
    assert_ne!(
        rederived.collapse_prevention_gates(),
        declared,
        "the moved root's own gates must differ from the declared set, or this test \
         cannot tell holding the gates from re-deriving them"
    );

    OuterObjective::reset(&mut objective);
    assert_eq!(
        objective.collapse_prevention_gates.as_ref(),
        Some(&declared),
        "a reset starts another seed of the SAME objective and must keep its gates"
    );

    let contract = OuterObjective::reactive_domain_scalar_contract(&objective)
        .expect("the scalar contract must construct")
        .expect("a dense K=2 objective advertises a reactive scalar contract");
    OuterObjective::begin_reactive_domain_waypoint(&mut objective)
        .expect("the objective must open a waypoint transaction");
    OuterObjective::install_reactive_domain_scalar_state(&mut objective, contract.target())
        .expect("the objective must install its literal target scalar state");
    assert!(
        objective.collapse_prevention_gates.is_none(),
        "a scalar waypoint installs another objective, whose first root chooses its gates"
    );
    OuterObjective::rollback_reactive_domain_waypoint(&mut objective)
        .expect("the waypoint must roll back");
    assert_eq!(
        objective.collapse_prevention_gates.as_ref(),
        Some(&declared),
        "a rollback must restore the checkpointed objective's gates"
    );
}

/// A refresh on a routing where no atom pair co-fires freezes an EMPTY support.
/// Once declared, moving the logits until the pair co-fires must not turn the
/// separation barrier on: the gradient assembled under that support carries no
/// barrier force, so a live read would price a routing force the solve never
/// modelled. Positive control: re-deriving the support at the moved routing prices
/// a barrier.
#[test]
fn a_declared_empty_coactivation_support_stays_empty_when_routing_moves_2933() {
    let (mut term, _target, _rho) = co_firing_collinear_two_atom_fixture();
    let n = term.n_obs();
    // Every row routes to one atom, and the other atom sits far below the co-firing
    // floor relative to that row's peak.
    term.assignment.logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        if (row + atom) % 2 == 0 { 8.0 } else { -8.0 }
    });
    term.refresh_decoder_repulsion_gate();
    term.refresh_barrier_coactivation_gate();
    term.refresh_amplitude_barrier_gate();
    let disjoint = term.collapse_prevention_gates();
    assert!(
        disjoint
            .barrier_coactivation
            .as_ref()
            .is_none_or(|gate| gate.pairs.is_empty()),
        "the disjoint routing must have no co-firing pair: {disjoint:?}"
    );
    term.declare_collapse_prevention_gates(&disjoint);
    let before = term.separation_barrier_value(1.0);

    term.assignment.logits.fill(0.0);
    let after = term.separation_barrier_value(1.0);
    let mut rederived = term.clone();
    rederived.refresh_barrier_coactivation_gate();
    let live = rederived.separation_barrier_value(1.0);
    eprintln!(
        "[#2933 F05] declared empty support: before={before:e} after={after:e} \
         rederived={live:e}"
    );
    assert!(
        live > 0.0,
        "positive control: the support re-derived at the co-firing routing must price a \
         barrier, got {live:e}"
    );
    assert_eq!(
        after.to_bits(),
        before.to_bits(),
        "a declared empty coactivation support must keep the separation barrier off while \
         the routing moves (before={before:e}, after={after:e})"
    );
}
