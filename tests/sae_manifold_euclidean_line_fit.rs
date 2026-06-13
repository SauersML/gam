//! Regression pin for #1051 — the euclidean-1D SAE atom must fit a straight
//! line trivially through the production outer engine.
//!
//! A straight line is the canonical euclidean-1D manifold. Planting one in a
//! `p = 3` ambient with a degree-2 euclidean patch (`basis = [1, t, t²]`) makes
//! the decoder design *rank-deficient*: only the linear column carries signal,
//! so the joint Hessian has a genuine near-null direction in the β (decoder)
//! block, OUTSIDE the closed-form chart gauge orbit. Before the fix this made
//! `outer_gradient_conditioning_error` report "analytic outer gradient
//! undefined (joint Hessian numerically singular)" at the continuation spine,
//! every seed was rejected after ~20 s, the inner solve never converged, and
//! the outer BFGS terminated at the `1e12` infeasible sentinel.
//!
//! This test drives the fit *exactly the way production does* — the generic
//! outer cascade (`OuterProblem::run`) around `SaeManifoldOuterObjective`, the
//! same engine `gam-pyffi`'s `sae_manifold_fit` drives — and asserts the fit
//! (a) returns a finite criterion (NOT the `1e12` sentinel) and (b) recovers
//! the line to high reconstruction R².

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::outer_strategy::OuterProblem;
use gam::terms::latent_coord::LatentManifold;
use gam::terms::sae_manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const N: usize = 220;
const P: usize = 3;
const MAX_DEGREE: usize = 2; // basis [1, t, t²] ⇒ n_basis = 3 (issue's "rank=k/3")
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 60;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Deterministic index-keyed standard-normal-ish noise (no RNG dependency).
fn idx_noise(seed: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((s >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    let mut s2 = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s2 = s2
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u2 = ((s2 >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    (u + u2 - 1.0) * std::f64::consts::SQRT_2
}

/// A straight line `z = a + s·d` in `R^P` with the scalar latent `s` uniform on
/// [-1.5, 1.5], plus small per-coordinate noise. Returns `(z, s_true)`.
fn planted_line() -> (Array2<f64>, Vec<f64>) {
    let direction = [1.0_f64, 0.6, -0.3];
    let offset = [0.2_f64, -0.1, 0.4];
    let sigma = 0.03;
    let mut z = Array2::<f64>::zeros((N, P));
    let mut s_true = Vec::with_capacity(N);
    for row in 0..N {
        // Deterministic uniform on [-1.5, 1.5].
        let u = ((row as f64) + 0.5) / (N as f64);
        let s = -1.5 + 3.0 * u;
        s_true.push(s);
        for col in 0..P {
            z[[row, col]] =
                offset[col] + s * direction[col] + sigma * idx_noise((row as u64) * 7 + col as u64);
        }
    }
    (z, s_true)
}

/// Closed-form weighted-LSQ decoder init: `B = (ΦᵀΦ + εI)⁻¹ Φᵀ Z`, the cold
/// decoder the production seed hands the outer engine.
fn decoder_lsq_init(phi: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
    let m = phi.ncols();
    let mut gram = fast_ata(phi);
    for i in 0..m {
        gram[[i, i]] += 1.0e-8;
    }
    let rhs = fast_atb(phi, &z.to_owned());
    gram.cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&rhs)
}

/// Build the cold K=1 euclidean-line term the production driver hands the
/// engine: latent coordinate seeded from the (offset) true scalar, degree-2
/// euclidean patch, closed-form LSQ decoder.
fn build_cold_term(s_true: &[f64], z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, MAX_DEGREE).unwrap());
    let n_basis = evaluator.basis_size();
    // Seed coordinates from the true scalar with a small offset (coordinate
    // recovery is not under test; the rank deficiency is).
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| s_true[i] + 0.05);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let decoder = decoder_lsq_init(&phi, z);
    let atom = SaeManifoldAtom::new(
        "line_0",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(n_basis),
    )
    .unwrap()
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

fn dimensionless_entry_rho(term: &SaeManifoldTerm, z: &Array2<f64>) -> SaeManifoldRho {
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    assert!(seed_dispersion.is_finite() && seed_dispersion > 0.0);
    // Native ARD ON with one precision per latent axis — exactly what the
    // production `sae_manifold_fit` seeds (`native_ard_enabled = true` ⇒
    // `log_ard = zeros(d)`). The coordinate Gaussian prior this installs
    // regularises the otherwise-unbounded euclidean latent `t`, which is what
    // keeps the joint (coord × decoder) inner solve well-conditioned.
    SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(1)],
    )
    .seed_scaled_by_dispersion(seed_dispersion)
    .expect("dimensionless seed scaling")
}

/// Reconstruction R² over all rows (1 ⇒ perfect recovery of the line).
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
fn sae_manifold_euclidean_line_fits_through_production_engine() {
    let (z, s_true) = planted_line();
    let term = build_cold_term(&s_true, &z);
    let init_rho = dimensionless_entry_rho(&term, &z);
    let init_rho_flat = init_rho.to_flat();
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
        .run(&mut objective, "SAE euclidean line fit (#1051)")
        .expect("outer cascade must complete");
    let (fitted_term, _rho, _loss) = objective.into_fitted();
    let fitted = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted, &z);

    println!(
        "[#1051] euclidean line fit: final_value={:.6e} recon_R2={:.6}",
        result.final_value, r2
    );

    // (a) The outer engine must NOT terminate at the 1e12 infeasible sentinel:
    // the rank-deficient β-null direction must be deflated in the outer
    // gradient, not left to reject every continuation seed.
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "euclidean line fit terminated at the infeasible sentinel \
         (final_value={:.6e}); the rank-deficient decoder β-null was not \
         deflated in the outer gradient",
        result.final_value
    );

    // (b) A straight line is trivially representable by the linear column of the
    // euclidean patch — the fit must recover it.
    assert!(
        r2 > 0.99,
        "euclidean line reconstruction R²={r2:.6} < 0.99 — the canonical \
         euclidean-1D manifold was not recovered"
    );
}

/// Fast diagnostic: time a few raw inner Newton iterations at the cold entry ρ.
/// Isolates the inner `run_joint_fit_arrow_schur` solve from the continuation
/// walk and the undamped-evidence refine loop, so a per-iteration grind on the
/// rank-deficient β block surfaces in seconds. Each of `n` iterations should be
/// cheap; a single iteration taking many seconds localises the cost.
#[test]
fn sae_manifold_euclidean_line_inner_solve_iterations_are_cheap() {
    let (z, s_true) = planted_line();
    let mut term = build_cold_term(&s_true, &z);
    let mut rho = dimensionless_entry_rho(&term, &z);
    // Convergence-horizon trace: how the inner loss settles with budget.
    for budget in [20usize, 20, 20, 40, 100] {
        let t0 = std::time::Instant::now();
        let loss = term
            .run_joint_fit_arrow_schur(
                z.view(),
                &mut rho,
                None,
                budget,
                LEARNING_RATE,
                RIDGE_EXT_COORD,
                RIDGE_BETA,
            )
            .expect("inner solve must not error on the line");
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "[#1051-probe] inner +{budget}it total={:.8e} dt={dt:.3}s",
            loss.total()
        );
        assert!(loss.total().is_finite(), "inner loss non-finite");
    }

    // The real gate: a full reml_criterion (the inner solve PLUS the
    // undamped-evidence refine loop the continuation calls per ρ) must return a
    // finite, rankable criterion quickly — not grind to the 1e12 sentinel.
    let mut term2 = build_cold_term(&s_true, &z);
    let rho2 = dimensionless_entry_rho(&term2, &z);
    let t0 = std::time::Instant::now();
    let outcome = term2.reml_criterion(
        z.view(),
        &rho2,
        None,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    let dt = t0.elapsed().as_secs_f64();
    match &outcome {
        Ok((v, _)) => println!("[#1051-probe] reml_criterion OK value={v:.6e} dt={dt:.2}s"),
        Err(e) => println!("[#1051-probe] reml_criterion ERR dt={dt:.2}s: {e}"),
    }
    let (value, _) = outcome.expect("reml_criterion must succeed on the line");
    assert!(
        value.is_finite() && value < 1.0e11,
        "reml_criterion returned the infeasible sentinel value={value:.6e}"
    );
    assert!(
        dt < 30.0,
        "reml_criterion took {dt:.2}s (> 30s) — the inner solve grinds the rank-deficient block"
    );
}
