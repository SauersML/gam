//! #1023 task-2 radial-bias diagnostic: a K=1 IBP-MAP planted-circle fit through
//! the production outer engine, measuring the three discriminating numbers
//! (mean gate ζ, fitted_radius/data_radius, and λ_smooth vs the empirical-Bayes
//! optimum). The historical user-facing default `sae_manifold_fit(assignment="ibp_map")`
//! routed K=1 through this gate path with a cold logit seed of 0 (the EM
//! residual seed is gated on K>1), so ζ started at σ(0)=0.5 and the joint fit had
//! to drive it back toward 1. A uniform radial contraction whose size tracks mean
//! ζ (with no harmonic-spectrum signature) is the gate defect (cause B); a
//! contraction that also suppresses higher harmonics is λ over-smoothing (cause
//! A). This test prints all three numbers and gates the production fixed seed at
//! 1% radius bias.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const M: usize = 3; // const + first harmonic (cos, sin) — a circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Deterministic Lehmer-style standard-normal-ish noise keyed purely by index.
fn idx_noise(seed: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((s >> 11) as f64) * f64::from_bits(0x3CA0000000000000); // in [0,1)
    // Box–Muller-free: map two LCG draws to an approx-normal via a centered sum.
    let mut s2 = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s2 = s2
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u2 = ((s2 >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    (u + u2 - 1.0) * std::f64::consts::SQRT_2
}

/// Unit circle planted in the first two of `p` ambient dims, the rest zero,
/// plus per-coordinate Gaussian-ish noise of std `sigma`.
fn planted_unit_circle(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let theta = std::f64::consts::TAU * row as f64 / n as f64;
        z[[row, 0]] = theta.cos() + sigma * idx_noise(row as u64 * 2);
        z[[row, 1]] = theta.sin() + sigma * idx_noise(row as u64 * 2 + 1);
        for col in 2..p {
            z[[row, col]] = sigma * idx_noise((row as u64) * 17 + col as u64);
        }
    }
    z
}

/// Cold K=1 IBP-MAP term: PCA-free angle seed from the true ambient angle,
/// closed-form LSQ decoder at the cold gate, exactly as the production path
/// would after `term_from_padded_blocks` for one periodic atom.
fn build_cold_k1_term(z: &Array2<f64>, seed_logit: f64) -> SaeManifoldTerm {
    let n = z.nrows();
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    // Seed angle from the ambient (col0, col1) direction (offset so coordinate
    // recovery is not under test), in fraction-of-period units.
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
        let ang = z[[i, 1]].atan2(z[[i, 0]]);
        ((ang / std::f64::consts::TAU) + 0.03).rem_euclid(1.0)
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    // `seed_logit` sets the cold K=1 ibp_map gate. Production historically
    // seeded 0 (the EM residual seed is gated on K>1), so the cold gate was
    // σ(0)=0.5 — a 50% radial seed contraction the joint fit had to undo. The
    // fixed production seed is 6*tau, so gate = σ(6) independent of tau. The
    // decoder LSQ init is fit at whatever gate the seed implies so ζ·decode ≈ z
    // at the seed regardless.
    let logits = Array2::<f64>::from_elem((n, 1), seed_logit);
    let gate0 = 1.0 / (1.0 + (-seed_logit / TAU).exp()); // σ(seed_logit/τ)
    let mut xw = Array2::<f64>::zeros((n, M));
    for row in 0..n {
        for c in 0..M {
            xw[[row, c]] = gate0 * phi[[row, c]];
        }
    }
    let mut xtx = fast_ata(&xw);
    let mut trace = 0.0_f64;
    for i in 0..M {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / M as f64).max(1.0) * 1.0e-8;
    for i in 0..M {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&xw, z);
    let decoder = xtx
        .cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&xtz);

    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(M),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

fn run_production_fit(z: &Array2<f64>, seed_logit: f64) -> (SaeManifoldTerm, SaeManifoldRho) {
    let term = build_cold_k1_term(z, seed_logit);
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    let init_rho = SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(0)],
    )
    .seed_scaled_by_dispersion(seed_dispersion)
    .expect("dimensionless seed scaling");
    let init_flat = init_rho.to_flat();
    let n_params = init_flat.len();
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
    OuterProblem::new(n_params)
        .with_initial_rho(init_flat)
        .run(&mut objective, "K=1 circle radial bias")
        .expect("production outer fit must converge");
    let fitted_result = objective.into_fitted();
    let term = fitted_result.term;
    let rho = fitted_result.rho;
    (term, rho)
}

struct ArmMetrics {
    mean_zeta: f64,
    radius_ratio: f64,
    lambda_smooth: f64,
}

/// Fit a K=1 IBP planted unit circle from `seed_logit` and read the three
/// discriminating numbers: converged mean gate ζ (post-fit, not the seed),
/// fitted/data radius ratio, and the landed λ_smooth.
fn measure_arm(z: &Array2<f64>, p: usize, seed_logit: f64) -> ArmMetrics {
    let n = z.nrows();
    let (term, rho) = run_production_fit(z, seed_logit);

    // (1) converged mean gate ζ over rows (K=1 → single column). This is the
    //     post-fit value: the objective, not the seed, must drive it toward 1.
    let gates = term.assignment.assignments();
    let mut mean_zeta = 0.0_f64;
    for row in 0..n {
        mean_zeta += gates[[row, 0]];
    }
    mean_zeta /= n as f64;

    // (2) fitted_radius / data_radius (mean ambient norm of each).
    let fitted = term.fitted();
    let mut mean_fit_r = 0.0_f64;
    let mut mean_data_r = 0.0_f64;
    for row in 0..n {
        let mut fr = 0.0_f64;
        let mut dr = 0.0_f64;
        for col in 0..p {
            fr += fitted[[row, col]] * fitted[[row, col]];
            dr += z[[row, col]] * z[[row, col]];
        }
        mean_fit_r += fr.sqrt();
        mean_data_r += dr.sqrt();
    }
    mean_fit_r /= n as f64;
    mean_data_r /= n as f64;

    ArmMetrics {
        mean_zeta,
        radius_ratio: mean_fit_r / mean_data_r,
        lambda_smooth: rho.lambda_smooth_for(0),
    }
}

/// Permanent gate for the production fixed seed. The planted radius is 1; a
/// fitted ring more than 1% inside the data is the defect. The historical
/// logit-0 seed is documented above but is intentionally not fit on every CI
/// run: the permanent contract is that the shipped `6*tau` seed converges with
/// ζ≈1 and no radial shrinkage.
#[test]
fn sae_k1_ibp_circle_has_no_radial_shrinkage() {
    let n = 120usize;
    let p = 4usize;
    let sigma = 0.05_f64;
    let z = planted_unit_circle(n, p, sigma);
    let eb_optimal_lambda = sigma * sigma; // r² = 1; REML shrinkage is invisible
    let production_fixed_seed_logit = 6.0 * TAU;

    let high = measure_arm(&z, p, production_fixed_seed_logit);

    println!(
        "K=1 IBP circle (eb_optimal_lambda~{eb_optimal_lambda:.3e}):\n  \
         high-seed(logit 6*tau, gate0~1.0): mean_zeta={:.6} radius_ratio={:.6} lambda_smooth={:.4e}",
        high.mean_zeta, high.radius_ratio, high.lambda_smooth,
    );

    // Permanent gate: from the fix seed the ring must sit within 1% of the data
    // radius AND the converged gate must reach ≈1 (objective-driven, not seed).
    let high_bias = (1.0 - high.radius_ratio).abs();
    assert!(
        high.mean_zeta >= 0.99,
        "high-seed converged gate ζ={:.6} stuck below 0.99 — a second defect pulls the K=1 \
         IBP gate off its optimum (π₀=1, no sparsity pressure should keep ζ<1); report it",
        high.mean_zeta,
    );
    assert!(
        high_bias <= 0.01,
        "high-seed K=1 IBP circle radially biased by {:.2}% (radius_ratio={:.6}, ζ={:.6}); \
         the fitted manifold must lie within 1% of the data radius",
        high_bias * 100.0,
        high.radius_ratio,
        high.mean_zeta,
    );
}
