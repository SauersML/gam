//! Regression pin for #1095 — a circle (K=1, d=2 periodic) SAE atom must accept
//! at least one seed and fit on a SMALL activation bank (N=180), the same size
//! as the OLMo L44 color bank where every candidate seed was rejected at the
//! curvature-homotopy entry validation stage ("all 13 seeds rejected").
//!
//! The issue signature: on N=180 the joint Hessian at the early curvature spine
//! has a sub-floor pivot (`min pivot 3.865e-9`) that the entry walk classified
//! as a NON-gauge branch bifurcation, refusing the seed; the SAME settings on
//! N=635 converge. The circle's rotation gauge null must be recognised so a
//! small bank fits instead of erroring with `RemlConvergenceError`.
//!
//! The term and its initial ρ come from the production seed builders
//! (`build_sae_minimal_seed`, then `build_sae_fit_seed`), the route the Python
//! `sae_manifold_fit` entry takes, and the test drives the outer cascade
//! (`OuterProblem::run` around `SaeManifoldOuterObjective`). It asserts the
//! cascade COMPLETES with a finite criterion (not the all-seeds-rejected startup
//! error) and recovers the circle.
//!
//! TRACKED RED (#2822, the bounded circle marginal's acceptance case): the
//! criterion prefers the collapsed chart. With ARD held at the outer run's
//! checkpoint (`log α = −14.37`), raising `log λ_smooth` from 0 to 5 to 10 moves
//! the decoder's `‖B‖²` from 7.39 to 0.504 to 1.8e-3 and the data fit from 8.82
//! to 53.9 to 98.6, yet the criterion falls from 418.6 to 332.7 to −99.8. The
//! outer search follows that fall to `log λ_smooth = 23.2`, where the inner solve
//! stalls (`‖g‖ = 3.14` against tolerance 9.3e-5), so the terminal value-only
//! certificate is `+∞` and nothing is minted (sae-finish probe 1338781 at
//! `34ea9c6ca2`). ad-efs's design (#2822 comment 5721357089) attributes the
//! collapse preference to an unbounded per-row `½log κ` reward, which the bounded
//! von Mises circle marginal removes; this pin is that marginal's acceptance case.

use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::analytic_penalties::AnalyticPenaltyRegistry;
use gam::terms::sae::manifold::{
    SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest,
    SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm, SaeMinimalSeedRequest,
    build_sae_fit_seed, build_sae_minimal_seed,
};
use ndarray::Array2;

const N: usize = 180; // the L44 color-bank size from #1095
const P: usize = 24; // PCA ambient (issue uses 32; 24 keeps the test cheap)
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Deterministic Lehmer-style uniform in [0,1) keyed by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
}

fn idx_normal(seed: u64) -> f64 {
    let u1 = idx_uniform(seed).max(1.0e-12);
    let u2 = idx_uniform(seed.wrapping_add(0x9E3779B97F4A7C15));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A circle planted in a 2-plane of `R^P`, plus small ambient noise. This is a
/// genuinely-circular bank (so a fit exists); the bug is purely the small-N
/// seed-acceptance gate, not representability.
fn planted_small_circle() -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let theta = std::f64::consts::TAU * ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        // plane spanned by ambient axes 0 and 1, mild signal on a few more.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos();
        for col in 0..P {
            z[[i, col]] += 0.03 * idx_normal((i as u64) * 31 + col as u64);
        }
    }
    z
}

/// The cold K=1 circle term and its initial ρ, built by the production seed
/// builders: `build_sae_minimal_seed` seeds the latent angles from the bank's PCA
/// plane and least-squares-fits the decoder to the data, and `build_sae_fit_seed`
/// assembles the term. #2822: this fixture used to hand-build an all-zero decoder
/// with all-zero routing logits, which is not a start production hands the outer
/// cascade.
fn build_production_seed(z: &Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let assignment_kind = SaeFitAssignmentKind::OrderedBetaBernoulli;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind,
        alpha: ALPHA,
        tau: TAU,
        threshold: 0.0,
        top_k: None,
        random_state: 0,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("the production minimal seed builds on the small-N bank");
    let registry = AnalyticPenaltyRegistry::new();
    let SaeFitSeedReport {
        base_term,
        initial_rho,
        ..
    } = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: ALPHA,
        tau: TAU,
        learnable_alpha: false,
        assignment_kind,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: INNER_MAX_ITER,
        learning_rate: LEARNING_RATE,
        ridge_ext_coord: RIDGE_EXT_COORD,
        ridge_beta: RIDGE_BETA,
        top_k: None,
        threshold: 0.0,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 0,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("the production fit seed builds on the small-N bank");
    (base_term, initial_rho)
}

fn reconstruction_r2(fitted: &Array2<f64>, z: &Array2<f64>) -> f64 {
    let mut zbar = 0.0;
    for v in z.iter() {
        zbar += *v;
    }
    zbar /= (N * P) as f64;
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for (fi, zi) in fitted.iter().zip(z.iter()) {
        ssr += (fi - zi) * (fi - zi);
    }
    for v in z.iter() {
        sst += (v - zbar) * (v - zbar);
    }
    1.0 - ssr / sst.max(1.0e-300)
}

#[test]
fn sae_manifold_small_n_circle_accepts_a_seed_and_fits() {
    let z = planted_small_circle();
    let (term, init_rho) = build_production_seed(&z);
    let init_rho_flat = init_rho
        .to_flat(&term.assignment)
        .expect("the seed rho is bound to the term's assignment");
    let n_params = init_rho_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    let problem = OuterProblem::new(n_params).with_initial_rho(init_rho_flat);
    let result = problem
        .run(&mut objective, "SAE small-N circle seed accept (#1095)")
        .expect(
            "outer cascade must complete on a small-N circle bank — \
             all seeds rejected reproduces #1095",
        );
    objective
        .certify_outer_result(&result)
        .expect("small-N circle outer result must certify the installed state");
    let fitted_term = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    let fitted = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted, &z);
    println!(
        "[#1095] small-N circle fit: final_value={:.6e} recon_R2={:.6}",
        result.final_value, r2
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "small-N circle fit terminated at the infeasible sentinel \
         (final_value={:.6e})",
        result.final_value
    );
    assert!(
        r2 > 0.9,
        "small-N circle reconstruction R²={r2:.6} < 0.9 — the circle was not recovered"
    );
}
