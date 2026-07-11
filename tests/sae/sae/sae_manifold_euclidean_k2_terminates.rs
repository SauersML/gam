//! Regression pin for #1094 — a K=2 euclidean SAE fit must terminate with a
//! convergence certificate on small real-ish activation geometry. The
//! issue: on PCA-whitened OLMo activations the K≥2 euclidean fit stalls after
//! the rank-deficiency audit and never makes an outer iteration (the inner
//! Arrow-Schur joint solve grinds the ill-conditioned multi-atom Hessian; the
//! continuation walk re-enters every leg without arriving). K=1 circle dodges
//! it; multi-atom euclidean is the primary dictionary-learning use case.
//!
//! This drives the fit exactly the way production does (`OuterProblem::run`
//! around `SaeManifoldOuterObjective`) on TWO planted euclidean-1D atoms in a
//! small ambient, and asserts the cascade returns a converged, finite fit.
//! Elapsed time is reported diagnostically but never controls success.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

use faer::Side as FaerSide;

const N: usize = 240;
const P: usize = 6;
const K: usize = 2;
const MAX_DEGREE: usize = 2; // basis [1, t, t²] per atom
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
fn idx_noise(seed: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((s >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    (u - 0.5) * 2.0
}

/// Two planted lines in mutually disjoint ambient 2-planes; rows split
/// 50/50 by index so each atom is truly active on its half.
fn planted_two_lines() -> (Array2<f64>, [Vec<f64>; 2], Vec<usize>) {
    // atom A direction in axes (0,1); atom B direction in axes (2,3).
    let dir = [
        [1.0_f64, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.5, 0.0, 0.0],
    ];
    let off = [
        [0.2_f64, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.1, 0.0, 0.0],
    ];
    let sigma = 0.03;
    let mut z = Array2::<f64>::zeros((N, P));
    let mut s_true = [vec![0.0; N], vec![0.0; N]];
    let mut owner = vec![0usize; N];
    for row in 0..N {
        let k = row % 2;
        owner[row] = k;
        let u = ((row as f64) + 0.5) / (N as f64);
        let s = -1.5 + 3.0 * u;
        s_true[k][row] = s;
        for col in 0..P {
            z[[row, col]] =
                off[k][col] + s * dir[k][col] + sigma * idx_noise((row as u64) * 7 + col as u64);
        }
    }
    (z, s_true, owner)
}

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

fn build_cold_k2_term(s_true: &[Vec<f64>; 2], owner: &[usize], z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, MAX_DEGREE).unwrap());
    let n_basis = evaluator.basis_size();
    let mut atoms = Vec::with_capacity(K);
    let mut coords_all = Vec::with_capacity(K);
    let mut manifolds = Vec::with_capacity(K);
    for k in 0..K {
        let coords = Array2::from_shape_fn((N, 1), |(i, _)| {
            if owner[i] == k {
                s_true[k][i] + 0.05
            } else {
                0.0
            }
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let decoder = decoder_lsq_init(&phi, z);
        let atom = SaeManifoldAtom::new(
            format!("line_{k}"),
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(n_basis),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coords_all.push(coords);
        manifolds.push(LatentManifold::Euclidean);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, K)),
        coords_all,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
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
fn sae_manifold_euclidean_k2_fit_terminates() {
    let (z, s_true, owner) = planted_two_lines();
    let term = build_cold_k2_term(&s_true, &owner, &z);
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); K]);
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
    let t0 = Instant::now();
    let result = problem
        .run(&mut objective, "SAE euclidean K=2 terminates (#1094)")
        .expect("outer cascade must complete on a K=2 euclidean fit");
    let elapsed = t0.elapsed().as_secs_f64();
    objective
        .certify_outer_result(&result)
        .expect("euclidean K=2 outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let mut fitted_term = fitted.term;
    let fitted_out = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted_out, &z);
    // #1094 residual — evaluate the OUTER criterion DIRECTLY at the converged
    // optimum, on the fit's own settled ρ, through the same public
    // `penalized_quasi_laplace_criterion` path the outer cascade uses. The residual bug was the
    // criterion refusing a feasible euclidean K=2 fit to the infeasible sentinel
    // (`1e12`): at the rank-deficient optimum the KKT gradient parks in the
    // weakly-identified decoder/gauge directions, so no stationarity certificate
    // fires even though the penalised objective is at its numerical floor, and the
    // driver returned the "did not converge" refusal instead of ranking the finite
    // deflated Laplace evidence. This recheck pins the criterion PATH itself — not
    // just the optimizer's reported bookkeeping value — returning a FINITE score
    // for a fit that reconstructs the two planted lines. A criterion that reports
    // infeasible for a demonstrably good fit poisons outer model comparison.
    let (recheck_cost, _loss) = fitted_term
        .penalized_quasi_laplace_criterion(
            z.view(),
            &fitted.rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("outer criterion must EVALUATE (not error/refuse) at the converged K=2 optimum");
    println!(
        "[#1094] euclidean K=2 fit: final_value={:.6e} recheck_criterion={:.6e} recon_R2={:.6} elapsed={elapsed:.1}s",
        result.final_value, recheck_cost, r2
    );
    assert!(result.converged, "run() must return only a certified fit");
    assert!(
        result.converged_via.is_some(),
        "a returned fit must name its convergence certificate"
    );
    // The fit reconstructs the two planted lines — a FEASIBLE fit by its own
    // reconstruction certificate.
    assert!(
        r2 > 0.9,
        "euclidean K=2 reconstruction R²={r2:.6} < 0.9 — the two lines were not recovered"
    );
    // Neither the optimizer's reported value NOR a fresh criterion evaluation at
    // the converged ρ may land on the infeasible sentinel for this feasible fit:
    // the criterion must be finite and consistent with the fit's quality (#1094).
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "euclidean K=2 fit terminated at the infeasible sentinel (final_value={:.6e})",
        result.final_value
    );
    assert!(
        recheck_cost.is_finite() && recheck_cost < 1.0e11,
        "euclidean K=2 outer criterion re-evaluated to the infeasible sentinel at the converged \
         optimum (recheck_criterion={recheck_cost:.6e}) despite R²={r2:.6} — the criterion reports \
         a demonstrably good fit as infeasible (#1094)"
    );
}
