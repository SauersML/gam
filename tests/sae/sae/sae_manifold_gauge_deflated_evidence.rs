use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::{rho_optimizer::OuterProblem, seeding::SeedConfig};
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::sae_pca_seed_initial_coords;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, ArrayView2};
use std::sync::Arc;

const N: usize = 150;
const M: usize = 3;

fn deterministic_circle_frame(p: usize) -> (Array1<f64>, Array1<f64>) {
    let mut u = Array1::<f64>::zeros(p);
    let mut v = Array1::<f64>::zeros(p);
    for col in 0..p {
        let x = col as f64 + 1.0;
        u[col] = (0.017 * x).sin() + 0.3 * (0.071 * x).cos();
        v[col] = (0.031 * x + 0.4).cos() - 0.2 * (0.053 * x).sin();
    }
    let u_norm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for value in u.iter_mut() {
        *value /= u_norm;
    }
    let uv = u.dot(&v);
    for col in 0..p {
        v[col] -= uv * u[col];
    }
    let v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for value in v.iter_mut() {
        *value /= v_norm;
    }
    (u, v)
}

fn planted_circle_data(p: usize) -> Array2<f64> {
    let (u, v) = deterministic_circle_frame(p);
    Array2::from_shape_fn((N, p), |(row, col)| {
        let theta = std::f64::consts::TAU * row as f64 / N as f64;
        theta.cos() * u[col] + theta.sin() * v[col]
    })
}

fn global_ev(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let (n, p) = target.dim();
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        for row in 0..n {
            means[col] += target[[row, col]];
        }
        means[col] /= n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let r = target[[row, col]] - fitted[[row, col]];
            ssr += r * r;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    1.0 - ssr / sst.max(1.0e-300)
}

fn planted_circle_term(z: ArrayView2<'_, f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(M).expect("periodic evaluator"));
    let seed_coords =
        sae_pca_seed_initial_coords(z, &[SaeAtomBasisKind::Periodic], &[1]).expect("PCA seed");
    let coords = seed_coords.slice(ndarray::s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic basis");
    let mut xtx = fast_ata(&phi);
    for diag in 0..xtx.nrows() {
        xtx[[diag, diag]] += 1.0e-10;
    }
    let z_owned = z.to_owned();
    let xtz = fast_atb(&phi, &z_owned);
    let decoder = xtx
        .cholesky(FaerSide::Lower)
        .expect("seed normal equation")
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
    .expect("atom")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((z.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    SaeManifoldTerm::new(vec![atom], assignment).expect("term")
}

fn run_outer_fit(term: SaeManifoldTerm, z: &Array2<f64>, label: &str) -> SaeManifoldTerm {
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![Array1::zeros(1)]);
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    let mut seed_config = SeedConfig::default();
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 0, 0.04, 1.0e-6, 1.0e-6);
    OuterProblem::new(n_params)
        .with_seed_config(seed_config)
        .with_initial_rho(init_rho_flat)
        .with_max_iter(1)
        .run(&mut objective, label)
        .expect("outer engine must complete");
    let fitted = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    fitted
}

#[test]
fn gauge_deflated_evidence_planted_circle_fits_high_p() {
    for p in [512usize, 2048] {
        let z = planted_circle_data(p);
        let term = planted_circle_term(z.view());
        let fitted = run_outer_fit(term, &z, &format!("gauge-aware H_tt evidence p={p}"));
        let ev = global_ev(z.view(), fitted.fitted().view());
        assert!(
            ev >= 0.95,
            "planted K=1 circle at p={p}, n={N} must fit to EV >= 0.95, got {ev:.6}"
        );
    }
}
