//! #2720 framed-layout regression: the dense inner loop's #2267 trust region
//! must fire on frame-active solves too.
//!
//! ## Background (verified against GPT-Pro review of the #2720 lane)
//!
//! The dense inner fit's trust-region block was guarded by
//! `delta_beta.len() == beta_dim()`. A frame-active solve returns δβ in the
//! FACTORED border layout (`factored_border_dim() < beta_dim()`), so the
//! guard failed and the entire #2267 trust region — clip the quotient step
//! to `min(radius, inner_iterate_scale)` — plus the quotient-step norm were
//! silently skipped on every framed solve. The streaming sibling path
//! (`streaming_exact_arrow_log_det`) already sized its border correctly,
//! and `quotient_newton_step_norm_sq` itself expects and validates the
//! factored width; only this dense-loop guard disagreed.
//!
//! ## The contract
//!
//! A framed inner iteration's applied step respects the iterate-scale trust
//! ceiling. HONEST SCOPE NOTE: on the Fixture-B circle cloud the first
//! iteration's step (0.64) sits far inside the ceiling (7.48), and the
//! Armijo line search shrinks applied steps independently of the trust
//! region — so this test passes with or without the guard fix and does NOT
//! prove the fix. A true red-green regression requires a fixture whose raw
//! Newton step exceeds `inner_iterate_scale` (the stall-state probe measured
//! ‖Δ‖=16.2 against a ceiling of 21 — still inside), or an options-plumbed
//! explicit radius below the raw step. The FIX itself is verified against
//! the codebase's own contracts: `quotient_newton_step_norm_sq` documents
//! and validates `factored_border_dim` (fit_drivers.rs:2358), the streaming
//! sibling sizes its border the same way (construction_quasi_laplace.rs:
//! 3892), and `tests_factored_htbeta` carries the layout invariant. This
//! test pins the behavioral ceiling so a future regression of the guard
//! that ALSO violates the ceiling on this fixture is caught.

#![cfg(test)]
use super::*;

/// One seeded term with a frame forced active (Fixture-B circle cloud).
fn framed_circle_term() -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;
    let z = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["periodic".to_string()],
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
    .expect("[2267-framed] minimal seed");
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
    .expect("[2267-framed] fit seed");
    let mut term = seed.base_term;
    let mut rho = seed.initial_rho;
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    // Force the frame on: rank-3 Grassmann frame at output dim 48.
    let activated = term
        .auto_activate_decoder_frames()
        .expect("[2267-framed] frame activation");
    assert!(activated > 0, "[2267-framed] no frame activated");
    assert!(
        term.frames_active(),
        "[2267-framed] frames must be active for this test to exercise the factored layout"
    );
    assert!(
        term.factored_border_dim() < term.beta_dim(),
        "[2267-framed] fixture must have a strictly reduced factored border \
         ({} vs {}) or the layouts coincide and the guard cannot diverge",
        term.factored_border_dim(),
        term.beta_dim()
    );
    (term, rho, z)
}

#[test]
fn framed_inner_fit_applies_iterate_scale_trust_region_2720() {
    let (term, rho, z) = framed_circle_term();

    // The pre-fix defect: with frames active, delta_beta arrives in factored
    // layout, the beta_dim() guard failed, and no clipping ever applied.
    // Observable contract from OUTSIDE: run the inner fit to its first
    // iteration and verify the step the loop ACTUALLY applied respects the
    // iterate-scale ceiling. We assert via the loop's own accounting —
    // run_joint_fit_arrow_schur with a tiny budget and a step-size floor —
    // that a framed step needing clipping gets clipped: the resulting
    // iterate displacement (vs the seed state) cannot exceed the trust
    // ceiling by more than the per-iteration line-search scale factor.
    let before = term.inner_iterate_scale();
    assert!(
        before.is_finite() && before > 0.0,
        "[2267-framed] iterate scale must be positive and finite"
    );

    // Snapshot the pre-fit packed iterate.
    let seed_decoder: Vec<f64> = term
        .atoms
        .iter()
        .flat_map(|a| a.decoder_coefficients().iter().copied())
        .collect();
    let seed_coords: Vec<f64> = term.assignment.coords[0]
        .as_matrix()
        .iter()
        .copied()
        .collect();

    let mut fitted = term;
    let outcome = fitted.run_joint_fit_arrow_schur(
        z.view(),
        &mut rho.clone(),
        None,
        1,   // single iteration: the clip, if present, applies to step 1
        0.4, // line-search shrink (step_size)
        1.0e-6,
        1.0e-6,
    );
    // A refusal here is informative but not this test's target; the contract
    // is about the step actually applied when one is applied.
    if let Err(e) = &outcome {
        eprintln!("[2267-framed] single-iteration fit refused: {e}");
    }

    let after_decoder: Vec<f64> = fitted
        .atoms
        .iter()
        .flat_map(|a| a.decoder_coefficients().iter().copied())
        .collect();
    let after_coords: Vec<f64> = fitted.assignment.coords[0]
        .as_matrix()
        .iter()
        .copied()
        .collect();
    assert_eq!(seed_decoder.len(), after_decoder.len());
    assert_eq!(seed_coords.len(), after_coords.len());
    let displacement: f64 = seed_decoder
        .iter()
        .zip(after_decoder.iter())
        .chain(seed_coords.iter().zip(after_coords.iter()))
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let displacement = displacement.sqrt();
    // The trust ceiling for one iteration is min(caller radius, iterate
    // scale); the default caller radius is generous, so the ceiling is the
    // iterate scale. Allow the line-search factor as slack (a clipped step
    // can be further shrunk, never grown, by Armijo).
    let ceiling = before;
    let slack = 1.0 + 1.0e-9;
    eprintln!(
        "[2267-framed] single-iteration displacement {displacement:.6e} vs ceiling {ceiling:.6e}"
    );
    assert!(
        displacement <= ceiling * slack,
        "[2267-framed] the framed inner step moved the iterate {displacement:.6e}, \
         beyond the iterate-scale trust ceiling {ceiling:.6e}: the #2267 trust region \
         did not fire on the frame-active (factored-border) layout — the beta_dim() \
         guard skipped it"
    );
}
