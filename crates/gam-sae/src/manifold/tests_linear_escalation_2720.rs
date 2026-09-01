//! Regression probes for the linear atom after the #2720 two-span fix.
//!
//! The geometry sweep found that the linear fixture reaches the same accepted
//! criterion at 40 and 400 iterations while the accepted point retains a
//! material derivative along the likelihood-flat dilation orbit. These tests
//! keep two observations pinned: the accepted criterion is budget-invariant,
//! and `descend_gauge_orbit` engages at the seed and reaches its own stopping
//! rule before exhausting its round budget.
//!
//! This module uses the planted-circle data only as a deterministic dataset.
//! Its geometry plan is linear; it is not a Poincaré-atom test.
//!
//! Run: `cargo test -p gam-sae linear_ -- --nocapture`

#![cfg(test)]
use super::*;
use crate::manifold::tests_gauge_posterior_flatness_2720::planted_circle_cloud;

/// ARD-saddle rho matching the #2772 sweep: log_lambda_sparse = −0.5,
/// log_lambda_smooth = −1.0, log_ard = −0.5; block channel left at seed value.
fn ard_saddle_rho(seed_rho: &SaeManifoldRho) -> SaeManifoldRho {
    let mut rho = seed_rho.clone();
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    rho
}

/// The linear/circle probe fixture: seeded term (d=1, unframed) + rho.
fn linear_circle() -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    let z = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["linear".to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("[linear-stall] minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("[linear-stall] fit seed");
    let mut term = seed.base_term;
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    (term, ard_saddle_rho(&seed.initial_rho), z)
}

/// Probe A: is the accepted linear criterion unchanged by a 10x budget
/// escalation?
#[test]
fn linear_budget_escalation_2720() {
    let mut exit_values: Vec<(usize, f64)> = Vec::new();
    for budget in [40usize, 400usize] {
        // Fresh deterministic seed per arm: the escalation must be measured
        // from the same start state, not the previous arm's terminal state.
        let (mut term, rho, z) = linear_circle();
        match term.penalized_quasi_laplace_criterion_with_cache(
            z.view(),
            &rho,
            None,
            budget,
            0.4,
            1.0e-6,
            1.0e-6,
        ) {
            Ok((value, loss, _cache)) => {
                exit_values.push((budget, value));
                eprintln!(
                    "[linear-stall] budget={budget}: SOLVED criterion={value:.6e} \
                     (data_fit={:.3e}, smoothness={:.3e}, ard={:.3e}, \
                     gauge_deflated_dirs={})",
                    loss.data_fit,
                    loss.smoothness,
                    loss.ard,
                    loss.criterion_gauge_deflated_directions
                );
            }
            Err(e) => {
                let note = e.to_string();
                eprintln!(
                    "[linear-stall] budget={budget}: REFUSED — {}",
                    note.chars().take(700).collect::<String>()
                );
            }
        }
    }

    // ENFORCED: the exit is budget-invariant — criterion bit-identical across
    // a 10x budget escalation (measured 1.560920e2 at both 40 and 400). The
    // 4.21x orbit slope the geometry sweep pins on linear is therefore a
    // property of the EXIT the machinery declares, not of truncation.
    assert!(exit_values.len() == 2, "both budget arms must solve");
    assert!(
        exit_values[0].1 == exit_values[1].1,
        "[linear-stall] the exit changed with budget ({} at {} vs {} at {}) — the \
         budget-invariance finding changed; re-measure before updating",
        exit_values[0].1,
        exit_values[0].0,
        exit_values[1].1,
        exit_values[1].0
    );
}

/// Probe B: does the orbit mover engage on the linear seed and finish under
/// the round bound? This distinguishes an inactive block from later exit
/// criterion behavior.
#[test]
fn linear_orbit_mover_engages_postfix_2720() {
    let (mut term, rho, z) = linear_circle();
    let registry = AnalyticPenaltyRegistry::new();
    let smooth = rho.lambda_smooth_vec().expect("[linear-stall] smooth vec");
    let descent = term
        .descend_gauge_orbit(z.view(), &rho, Some(&registry), &smooth, 40)
        .expect("[linear-stall] descend_gauge_orbit failed at the seed");
    eprintln!(
        "[linear-stall] orbit mover at seed: dim={} moved={} decrease={:.6e} \
         rounds={} max|gᵀvᵢ|={:.6e} evaluations={}",
        descent.dimension,
        descent.moved(),
        descent.objective_decrease,
        descent.rounds,
        descent.max_directional_derivative,
        descent.evaluations,
    );
    // ENFORCED: the mover ENGAGES on linear (moved, decrease > 0, rounds > 0)
    // and stops at its own stationarity BEFORE the round budget (11 < 40
    // measured) — while the geometry sweep still pins 4.21x of orbit slope at
    // the exit. The two stationarity bars (the mover's material floor and the
    // KKT gate's tolerance) are in different currencies; this pin keeps the
    // contrast measurable rather than anecdotal.
    assert!(
        descent.moved() && descent.objective_decrease > 0.0 && descent.rounds > 0,
        "[linear-stall] the orbit mover no longer engages on linear at the seed \
         (measured decrease=4.449559e0 over 11 rounds) — the engagement finding \
         changed; re-measure before updating"
    );
    assert!(
        descent.rounds < 40,
        "[linear-stall] the mover consumed its full round budget ({} rounds) at the \
         seed — it previously stopped at its own stationarity in 11; the stop \
         story changed",
        descent.rounds
    );
}
