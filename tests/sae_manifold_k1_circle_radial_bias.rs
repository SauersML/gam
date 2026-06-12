//! #1023 task-2 radial-bias diagnostic: a K=1 IBP-MAP planted-circle fit through
//! the production outer engine, measuring the three discriminating numbers
//! (mean gate ζ, fitted_radius/data_radius, and λ_smooth vs the empirical-Bayes
//! optimum). The user-facing default `sae_manifold_fit(assignment="ibp_map")`
//! routes K=1 through this gate path, where the cold logit seed is 0 (the EM
//! residual seed is gated on K>1) so ζ starts at σ(0)=0.5 and the joint fit must
//! drive it back toward 1. A uniform radial contraction whose size tracks mean ζ
//! (with no harmonic-spectrum signature) is the gate defect (cause B); a
//! contraction that also suppresses higher harmonics is λ over-smoothing
//! (cause A). This test prints all three numbers and gates the radius bias at 1%.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::outer_strategy::OuterProblem;
use gam::terms::latent_coord::LatentManifold;
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
fn build_cold_k1_term(z: &Array2<f64>) -> SaeManifoldTerm {
    let n = z.nrows();
    let p = z.ncols();
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    // Seed angle from the ambient (col0, col1) direction (offset so coordinate
    // recovery is not under test), in fraction-of-period units.
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
        let ang = z[[i, 1]].atan2(z[[i, 0]]);
        ((ang / std::f64::consts::TAU) + 0.03).rem_euclid(1.0)
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    // Cold K=1 ibp_map logit seed is 0 (production: EM residual seed gated on
    // K>1), so the cold gate is σ(0)=0.5. The decoder LSQ init is fit at that
    // gate so ζ·decode ≈ z at the seed.
    let logits = Array2::<f64>::zeros((n, 1));
    let gate0 = 0.5_f64; // σ(0/τ)
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

fn run_production_fit(z: &Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let term = build_cold_k1_term(z);
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    let init_rho = SaeManifoldRho::new(SPARSITY.ln(), SMOOTHNESS.ln(), vec![Array1::<f64>::zeros(0)])
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
    let (term, rho, _loss) = objective.into_fitted();
    (term, rho)
}

/// The three discriminating numbers + the 1% radial-bias gate. The planted
/// radius is 1; a fitted ring more than 1% inside the data is the defect.
#[test]
fn sae_k1_ibp_circle_has_no_radial_shrinkage() {
    let n = 250usize;
    let p = 8usize;
    let sigma = 0.05_f64;
    let z = planted_unit_circle(n, p, sigma);

    let (term, rho) = run_production_fit(&z);

    // (1) mean gate ζ over rows (K=1 → single column).
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
    let radius_ratio = mean_fit_r / mean_data_r;

    // (3) λ_smooth landed, against the empirical-Bayes optimum ~ σ²/r² (r=1):
    //     the REML-correct shrinkage is invisible (~2σ²/(n·r²)); anything that
    //     contracts the ring by % is a defect, not REML.
    let lambda_smooth = rho.lambda_smooth();
    let eb_optimal_lambda = sigma * sigma; // r² = 1

    println!(
        "K=1 IBP circle: mean_zeta={mean_zeta:.6} radius_ratio={radius_ratio:.6} \
         (fit_r={mean_fit_r:.5} data_r={mean_data_r:.5}) lambda_smooth={lambda_smooth:.4e} \
         eb_optimal~{eb_optimal_lambda:.4e} ratio_lambda={:.2e}",
        lambda_smooth / eb_optimal_lambda.max(1e-300),
    );

    // Discriminator: if radius_ratio ≈ mean_zeta the contraction is the gate
    // (cause B); if radius_ratio < mean_zeta the decoder is also over-smoothed
    // (cause A on top). Either way the ring must sit within 1% of the data.
    let radial_bias = (1.0 - radius_ratio).abs();
    assert!(
        radial_bias <= 0.01,
        "K=1 IBP circle ring is radially biased by {:.2}% (radius_ratio={radius_ratio:.6}, \
         mean_zeta={mean_zeta:.6}); the fitted manifold must lie within 1% of the data radius",
        radial_bias * 100.0,
    );
}
