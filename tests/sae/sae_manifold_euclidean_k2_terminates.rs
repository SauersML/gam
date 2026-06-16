//! Regression pin for #1094 — a K=2 euclidean SAE fit must TERMINATE (and
//! converge) within a bounded time on small real-ish activation geometry. The
//! issue: on PCA-whitened OLMo activations the K≥2 euclidean fit stalls after
//! the rank-deficiency audit and never makes an outer iteration (the inner
//! Arrow-Schur joint solve grinds the ill-conditioned multi-atom Hessian; the
//! continuation walk re-enters every leg without arriving). K=1 circle dodges
//! it; multi-atom euclidean is the primary dictionary-learning use case.
//!
//! This drives the fit exactly the way production does (`OuterProblem::run`
//! around `SaeManifoldOuterObjective`) on TWO planted euclidean-1D atoms in a
//! small ambient, and asserts the cascade COMPLETES with a finite criterion in
//! bounded wall-time — a hang reproduces #1094.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::outer_strategy::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
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
// Generous wall-clock ceiling: the K=2 euclidean fit must finish well inside
// this on N=240, P=6. A true hang blows past it (the issue's 240 s tile cap).
const WALL_CLOCK_CEILING_SECS: f64 = 150.0;

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
        AssignmentMode::ibp_map(TAU, ALPHA, false),
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
    let (fitted_term, _rho, _loss) = objective.into_fitted();
    let fitted = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted, &z);
    println!(
        "[#1094] euclidean K=2 fit: final_value={:.6e} recon_R2={:.6} elapsed={elapsed:.1}s",
        result.final_value, r2
    );
    assert!(
        elapsed < WALL_CLOCK_CEILING_SECS,
        "euclidean K=2 fit took {elapsed:.1}s > {WALL_CLOCK_CEILING_SECS:.0}s ceiling — \
         the multi-atom joint solve hangs (#1094)"
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "euclidean K=2 fit terminated at the infeasible sentinel (final_value={:.6e})",
        result.final_value
    );
    assert!(
        r2 > 0.9,
        "euclidean K=2 reconstruction R²={r2:.6} < 0.9 — the two lines were not recovered"
    );
}
