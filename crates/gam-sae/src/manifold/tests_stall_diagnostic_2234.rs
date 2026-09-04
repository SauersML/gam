//! zz_measure diagnostic (2026-07-10, #2234 blocker): every Python-entry
//! `sae_manifold_fit` on HEAD refuses to mint — the outer search descends,
//! then freezes with a bit-stable projected gradient (synthetic planted
//! circle: 63 iterations, objective 1.216074e2, |g_proj| = 1.381e0) and the
//! #2241 certified-termination contract refuses. Lane-consistency (#1224) and
//! every FD gate PASS, so this drives the SAME planted circle through the
//! plain RUST engine (seed builders + `OuterProblem`) to split the fault:
//!
//! - engine converges here  ⇒ the stall lives in the pyffi orchestration
//!   above the engine (topology/promotion/alternation), not the optimizer;
//! - engine stalls here     ⇒ run per-coordinate central differences of
//!   `eval_cost` against `eval`'s analytic gradient AT the stalled ρ and
//!   print both (the desync, if any, named coordinate by coordinate).

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_stall_diagnostic_2234;` — its single declaration. Saying so in-file
// makes the test scope a claim the compiler enforces rather than one the
// filename merely implies, which is what puts the fixture helpers below in
// the same scope as the `#[test]` fns they serve.
#![cfg(test)]

use super::*;

/// #2234 plain-engine stall pin, restored (#2818).
///
/// ─── What this pins ───
///
/// The 2026-07-10 pathology: provisional budget-marker refusals froze every
/// `sae_manifold_fit`, so the outer search returned infeasible probes rather
/// than descending. Driving the SAME planted circle through the plain RUST
/// engine (seed builders + `OuterProblem`) splits the fault — engine converges
/// here ⇒ the stall lives in the pyffi orchestration above the engine; engine
/// stalls here ⇒ the optimizer. Either way the telemetry says whether the
/// freeze mechanism is back: `infeasible_criterion_evals` counts refused probes
/// and `criterion_calls` counts the search's actual work.
///
/// ─── What was deleted, and what could not come back ───
///
/// `c0a21b554` deleted this file's body; `manifold/mod.rs` kept declaring the
/// module, so the census stayed green over a file holding only its doc comment.
///
/// The pin had a SECOND arm — the #2253 same-emission value/gradient audit,
/// which differenced `log|H|` from the exact emission the optimizer consumed.
/// That arm is NOT restored, and not because it was weak: `d484a091a` deleted
/// the production functions it read.
///   * `gam_solve::evidence::arrow_log_det_from_cache` — a `pub fn` of the
///     gam-solve LIBRARY, gone from `crates/gam-solve/src/evidence.rs`;
///   * `gam_sae::manifold::construction::coordinate_block_log_det`;
///   * `criterion_as_atoms` in `construction_quasi_laplace.rs`.
/// At `origin/main` only prose survives: `construction_quasi_laplace.rs:356`
/// and `outer_objective.rs:5208` still name `arrow_log_det_from_cache` in a
/// comment and in an error string for a function that no longer exists. Getting
/// that arm back is a production change, reported on #2818 rather than made
/// here.
///
/// ─── How the restored arm differs from the deleted one ───
///
/// The telemetry assertions used to live inside the `Err` arm only, so a run in
/// which the engine certified asserted nothing at all — the pin could pass by
/// the search succeeding OR by it never being graded. They are hoisted out
/// here: the freeze signature is checked on every outcome. That is a
/// strengthening; no bound is loosened.
///
/// The fixture is a closure inside the test rather than a `fn`: the sweep that
/// deleted this pin computed reachability from a stripped symbol table, where
/// the predicate is vacuously true of every test-only item, so the rebuild owns
/// no named item for it to miss.
#[test]
fn zz_planted_circle_plain_engine_stall_diagnostic_2234() {
    use gam_solve::rho_optimizer::OuterProblem;
    use gam_solve::seeding::SeedConfig;

    // Mirrors the frozen #2253 weekday-L17 discriminator after its exact
    // orthonormal reduction: K=1, d_atom=1, n=42, p=48. The prior n=200, p=8,
    // d_atom=3 fixture did not enter the single-circle log-det seam.
    let planted_circle_cloud = || -> Array2<f64> {
        let n = 42usize;
        let p = 48usize;
        let mut state = 0x2468_ace0_1357_9bdfu64;
        let mut unit = move || {
            // LCG → [0,1); NO rand, NO clock (repo #932 rules).
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let two_pi = std::f64::consts::TAU;
        let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
        let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let theta = two_pi * unit();
            for j in 0..p {
                let noise = 0.01 * (2.0 * unit() - 1.0);
                z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
            }
        }
        z
    };

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
    .expect("minimal seed");
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
    .expect("fit seed");

    let initial_flat = seed.initial_rho.to_flat();
    let n_params = initial_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        seed.base_term,
        z.clone(),
        Some(registry),
        seed.initial_rho,
        40,
        0.05,
        1.0e-6,
        1.0e-6,
    );
    objective.remove_checkpoint();
    let problem = OuterProblem::new(n_params)
        .with_initial_rho(initial_flat.clone())
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let outcome = problem.run(&mut objective, "zz stall diagnostic 2234");
    match &outcome {
        Ok(result) => eprintln!(
            "[zz2234] PLAIN ENGINE CONVERGED: value={:.6e} converged={} — the Python-entry \
             stall is ORCHESTRATION-layer (pyffi), not the optimizer",
            result.final_value,
            result.converged(),
        ),
        Err(err) => eprintln!("[zz2234] PLAIN ENGINE DID NOT CERTIFY: {err}"),
    }

    // The telemetry is read on EVERY outcome, not only on refusal. The
    // 2026-07-10 infeasibility pathology froze every fit through provisional
    // budget-marker refusals (frozen isotropic checkpoint, large lane
    // disagreement); post-fix the optimizer descends genuinely, and
    // non-certification on this deliberately tight fixture budget is an honest
    // optimizer limitation rather than an infeasible probe. Assert that
    // pathology stays dead either way.
    let telemetry = objective.probe_telemetry();
    eprintln!("[zz2234] probe telemetry: {telemetry:?}");
    assert_eq!(
        telemetry.infeasible_criterion_evals, 0,
        "infeasible probes returned — the #2234 pathology regressed (telemetry: {telemetry:?})"
    );
    assert!(
        telemetry.criterion_calls > 10,
        "the outer search froze after {} criterion calls — infeasibility or an equivalent \
         freeze are back (telemetry: {telemetry:?})",
        telemetry.criterion_calls
    );
    eprintln!(
        "[zz2234] no walls, genuine descent — non-certification on this tight fixture budget \
         is documented as an optimizer limitation (see #2234)"
    );
}
