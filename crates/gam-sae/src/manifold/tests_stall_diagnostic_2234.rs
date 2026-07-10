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
use super::*;

fn planted_circle_cloud() -> (Array2<f64>, usize) {
    // Mirrors the failing Python probe: n=200, p=8, rank-2 circle + 1e-2 noise.
    let n = 200usize;
    let p = 8usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        // LCG → [0,1); NO rand, NO clock (repo #932 rules).
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    (z, p)
}

#[test]
fn zz_planted_circle_plain_engine_stall_diagnostic_2234() {
    use gam_solve::rho_optimizer::OuterProblem;
    use gam_solve::seeding::SeedConfig;

    let (z, _p) = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![3],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        ibp_alpha_override: None,
        random_state: 7,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        atom_basis: &minimal.atom_basis,
        atom_dim: &minimal.effective_atom_dim,
        atom_centers: &minimal.atom_centers,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        basis_sizes: &minimal.basis_sizes,
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
        max_iter: 60,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 7,
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
        60,
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
    match outcome {
        Ok(result) => {
            eprintln!(
                "[zz2234] PLAIN ENGINE CONVERGED: value={:.6e} converged={} — the Python-entry \
                 stall is ORCHESTRATION-layer (pyffi), not the optimizer",
                result.final_value, result.converged,
            );
        }
        Err(err) => {
            eprintln!("[zz2234] PLAIN ENGINE STALLED: {err}");
            eprintln!(
                "[zz2234] probe telemetry: {:?}",
                objective.probe_telemetry()
            );
            // FD-vs-analytic at the banked best iterate: name the coordinate.
            let banked = objective
                .try_resume_from_checkpoint(n_params)
                .map(Array1::from)
                .unwrap_or(initial_flat);
            let grad = objective
                .eval(&banked)
                .expect("gradient eval at the stalled rho")
                .gradient;
            let h = 1.0e-3;
            for j in 0..n_params {
                let mut plus = banked.clone();
                plus[j] += h;
                let mut minus = banked.clone();
                minus[j] -= h;
                let c_plus = objective.eval_cost(&plus).expect("cost at +h");
                let c_minus = objective.eval_cost(&minus).expect("cost at -h");
                let fd = (c_plus - c_minus) / (2.0 * h);
                eprintln!(
                    "[zz2234] coord {j}: analytic={:+.6e} central_fd={:+.6e} |delta|={:.3e}",
                    grad[j],
                    fd,
                    (grad[j] - fd).abs()
                );
            }
            panic!("[zz2234] stall reproduced in the plain engine — see FD table above");
        }
    }
}
