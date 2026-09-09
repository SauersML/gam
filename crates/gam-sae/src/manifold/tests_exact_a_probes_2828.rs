//! Two probes of the exact-A θ-adjoint that the 2026-09-08 repair of
//! 14e1ce6d8 could not settle by reading (#2828):
//!
//! 1. the resident (row-jet Trace) softmax θ-adjoint against the dense one
//!    under BOTH evidence operators — the set-aside variant added a
//!    second-jet term to the row-jet β motion under `exact_a`, which is only a
//!    fix if the two routes currently disagree there;
//! 2. the β block of `exact_a_theta_adjoint_joint` against a central
//!    difference of `log|A|` in a decoder coefficient — by reading, the dense
//!    exact-A adjoint carries no third-derivative leg for the decoder priors'
//!    β–β curvature, and the t-block FD gate in `tests_logdet_adjoint_780`
//!    never probes β.
//!
//! Both print their full table; a failure is a measurement, not a verdict on
//! which route is right.
#![cfg(test)]
use super::*;
use crate::assignment::AssignmentMode;
use crate::manifold::arrow_solver::DeflatedArrowSolver;
use crate::manifold::construction::ThetaAdjointDhChannel;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use gam_solve::arrow_schur::{solve_arrow_newton_step_with_options, ArrowSolveOptions};
use ndarray::Array2;

fn frozen_anchor_and_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldTerm, ArrowFactorCache) {
    let mut anchor = term.clone();
    let (_value, _loss, cache) = anchor
        .penalized_quasi_laplace_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .expect("frozen-gate cache at the fixture state");
    anchor.streaming_gates_frozen = true;
    (anchor, cache)
}

/// `log|A|` at a state, with the gates pinned to the anchor's (the objective
/// whose Hessian `A` is differentiated holds them fixed).
fn frozen_exact_a_logdet(
    anchor: &SaeManifoldTerm,
    mut endpoint: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Option<f64> {
    endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
    endpoint.barrier_coactivation_gate = anchor.barrier_coactivation_gate.clone();
    endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
    endpoint.streaming_gates_frozen = true;
    let (_v, _l, cache) = endpoint
        .penalized_quasi_laplace_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .ok()?;
    endpoint
        .exact_observed_information_log_dets(rho, target.view(), &cache)
        .ok()
        .map(|(log_a, _)| log_a)
}

#[test]
fn resident_softmax_theta_adjoint_matches_dense_under_both_operators_2828() {
    let (mut term, target, rho) = threshold_gate_tiny_fixture(false);
    term.assignment.mode = AssignmentMode::softmax(0.8);
    let rho = rho.for_assignment(term.assignment.mode);
    // The softmax variant of this fixture is not a quasi-Laplace optimum, so
    // the cache is the plain arrow-Schur factorisation of the assembled
    // system at this state (the construction the set-aside variant's own
    // parity gate used), not a criterion cache.
    let mut anchor = term.clone();
    let mut system = anchor
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("cold arrow assembly at the softmax fixture state");
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
    let options = ArrowSolveOptions::direct();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("direct arrow-Schur factorisation at the softmax fixture state");
    anchor.streaming_gates_frozen = true;
    let solver = DeflatedArrowSolver::plain(&cache);
    let inverse = anchor
        .materialize_joint_inverse(&cache, &solver)
        .expect("dense joint inverse");
    let mut failures = Vec::new();
    for exact in [false, true] {
        let operator = if exact {
            EvidenceOperator::ExactObservedInformation
        } else {
            EvidenceOperator::Majorizer
        };
        let residual_target = exact.then_some(target.view());
        let dense = anchor
            .logdet_theta_adjoint_dense(
                &rho,
                &cache,
                &inverse,
                ThetaAdjointDhChannel::All,
                false,
                exact,
                residual_target,
            )
            .expect("dense theta adjoint");
        let resident = anchor
            .contracted_softmax_trace_adjoint(&rho, &cache, &solver, true, operator, residual_target)
            .expect("resident softmax trace adjoint");
        let scale = dense
            .t
            .iter()
            .chain(dense.beta.iter())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let gap_t = dense
            .t
            .iter()
            .zip(resident.t.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let gap_beta = dense
            .beta
            .iter()
            .zip(resident.beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "EXACT_A_PROBE_1 exact={exact} scale={scale:.3e} gap_t={gap_t:.3e} gap_beta={gap_beta:.3e}"
        );
        if gap_t.max(gap_beta) > 1.0e-9 * (1.0 + scale) {
            failures.push(format!("exact={exact}: t={gap_t:e} beta={gap_beta:e} scale={scale:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "resident and dense softmax θ-adjoints disagree beyond round-off: {failures:?}"
    );
}

#[test]
fn exact_a_joint_theta_adjoint_beta_block_matches_finite_difference_2828() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let (anchor, cache) = frozen_anchor_and_cache(&term, &target, &rho);
    let gamma = anchor
        .exact_a_theta_adjoint_joint(&rho, target.view(), &cache)
        .expect("analytic exact-A joint theta adjoint");
    let offsets = anchor.beta_offsets();
    let p = anchor.output_dim();
    let mut worst = 0.0_f64;
    let mut rows = Vec::new();
    for atom in 0..anchor.atoms.len().min(2) {
        let basis = anchor.atoms[atom].basis_size();
        let probes: Vec<(usize, usize)> = (0..basis)
            .flat_map(|mu| (0..p).map(move |out| (mu, out)))
            .take(6)
            .collect();
        for (mu, out) in probes {
            let index = offsets[atom] + mu * p + out;
            let analytic = gamma.beta[index];
            let mut estimates = [0.0_f64; 3];
            let coarse_step = 1.0e-4;
            for (step_index, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
                let h = coarse_step / divisor;
                let mut plus = anchor.clone();
                let mut minus = anchor.clone();
                for (endpoint, sign) in [(&mut plus, 1.0_f64), (&mut minus, -1.0)] {
                    let mut decoder = endpoint.atoms[atom].decoder_coefficients().clone();
                    decoder[[mu, out]] += sign * h;
                    endpoint.atoms[atom]
                        .set_decoder_coefficients(decoder)
                        .expect("same-shape decoder");
                }
                let a = frozen_exact_a_logdet(&anchor, plus, &target, &rho)
                    .expect("positive endpoint admitted");
                let b = frozen_exact_a_logdet(&anchor, minus, &target, &rho)
                    .expect("negative endpoint admitted");
                estimates[step_index] = (a - b) / (2.0 * h);
            }
            let coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
            let fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
            let oracle_error = (fine - coarse).abs() / (1.0 + fine.abs().max(coarse.abs()));
            let rel = (fine - analytic).abs() / (1.0 + fine.abs().max(analytic.abs()));
            eprintln!(
                "EXACT_A_PROBE_2 atom={atom} mu={mu} out={out} fd={fine:.9e} analytic={analytic:.9e} rel={rel:.3e} oracle_error={oracle_error:.3e}"
            );
            rows.push((atom, mu, out, fine, analytic, rel));
            worst = worst.max(rel + oracle_error);
        }
    }
    eprintln!("EXACT_A_PROBE_2 worst={worst:.3e} over {} β coordinates", rows.len());
    assert!(
        worst < 1.0e-3,
        "exact-A joint θ-adjoint β block must match FD: worst={worst:.3e}; rows={rows:?}"
    );
}
