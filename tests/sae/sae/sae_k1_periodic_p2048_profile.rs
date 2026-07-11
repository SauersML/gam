//! PROFILING harness (#1037 follow-up / K=1 perf lane): decompose the wall-time
//! of a K=1 periodic circle fit at real-bank scale (p=2048, multiple harmonics).
//!
//! The real banks (qwen p=2048, color p=5120) are K=1 PERIODIC circles. This
//! answers where a high-p K=1 periodic fit spends its time: the #1007 curvature-
//! homotopy walk (which for periodic only dials harmonics h>=2 and buys nothing
//! when the circle topology is already baked into the fundamental), versus the
//! full outer solve. Phase A times the walk alone; phase B times the full
//! production OuterProblem::run; B-A approximates the post-walk outer cost.
//! Prints a timed breakdown under `--nocapture`; the assertion only guards a
//! non-trivial fit (EV high), not timing.

use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::solver::seeding::SeedConfig;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::sae_pca_seed_initial_coords;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2, ArrayView2};
use std::sync::Arc;
use std::time::Instant;

// 7 cols = const + 3 harmonics: the fundamental traces the circle; h=2,3 are the
// curved columns the #1007 eta-dial actually scales, so the curvature walk does
// non-trivial work here (unlike the M=3 single-harmonic acceptance fixture).
const M: usize = 7;

fn planted_circle(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64) * 2.0
            - 1.0
    };
    let mut u = Array1::<f64>::from_shape_fn(p, |_| next());
    let mut v = Array1::<f64>::from_shape_fn(p, |_| next());
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    let uv = u.dot(&v);
    v.scaled_add(-uv, &u);
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        for col in 0..p {
            z[[row, col]] = c * u[col] + s * v[col] + 0.02 * next();
        }
    }
    z
}

fn build_term(z: ArrayView2<'_, f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap());
    let seed_coords =
        sae_pca_seed_initial_coords(z, &[SaeAtomBasisKind::Periodic], &[1]).expect("PCA seed");
    let coords = seed_coords.slice(ndarray::s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut xtx = fast_ata(&phi);
    for d in 0..xtx.nrows() {
        xtx[[d, d]] += 1.0e-9;
    }
    let z_owned = z.to_owned();
    let xtz = fast_atb(&phi, &z_owned);
    let decoder = xtx.cholesky(FaerSide::Lower).unwrap().solve_mat(&xtz);
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
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((z.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

fn global_ev(z: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let (n, p) = z.dim();
    let mut mean = Array1::<f64>::zeros(p);
    for row in 0..n {
        for col in 0..p {
            mean[col] += z[[row, col]];
        }
    }
    mean.mapv_inplace(|x| x / n as f64);
    let mut num = 0.0;
    let mut den = 0.0;
    for row in 0..n {
        for col in 0..p {
            let r = z[[row, col]] - fitted[[row, col]];
            num += r * r;
            let d = z[[row, col]] - mean[col];
            den += d * d;
        }
    }
    1.0 - num / den.max(1.0e-300)
}

fn make_objective(z: &Array2<f64>) -> (SaeManifoldOuterObjective, SaeManifoldRho) {
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![Array1::zeros(1)]);
    let obj = SaeManifoldOuterObjective::new(
        build_term(z.view()),
        z.clone(),
        None,
        rho.clone(),
        0,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    (obj, rho)
}

#[test]
fn profile_k1_periodic_high_p_phase_breakdown() {
    let n = 150;
    for &p in &[512usize, 2048usize] {
        let z = planted_circle(n, p, 0xBEEF ^ p as u64);

        // Phase A: the #1007 curvature-homotopy walk in isolation.
        let (mut obj_walk, _rho) = make_objective(&z);
        let t_a = Instant::now();
        let arrived = obj_walk.run_curvature_homotopy_entry();
        let walk_s = t_a.elapsed().as_secs_f64();

        // Phase C (gate-safety check): a COLD inner solve with NO walk, via
        // penalized_laml_criterion directly (the curvature walk lives only in the
        // OuterProblem seed loop). If this reaches EV >= 0.95 for K=1 periodic,
        // the curvature walk is pure overhead here and can be gated off — the
        // circle topology is baked into the fundamental harmonic, so the
        // Eckart-Young anchor is already on the circle (no linear basin to
        // escape).
        let mut term_cold = build_term(z.view());
        let cold_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![Array1::zeros(1)]);
        let t_c = Instant::now();
        let cold = term_cold.penalized_laml_criterion(z.view(), &cold_rho, None, 25, 1.0, 1.0e-6, 1.0e-6);
        let cold_s = t_c.elapsed().as_secs_f64();
        let cold_ev = if cold.is_ok() {
            global_ev(z.view(), term_cold.fitted().view())
        } else {
            f64::NAN
        };

        // Phase B: the full production outer fit (walk + cascade + outer rho).
        let (mut obj_full, rho) = make_objective(&z);
        let mut seed_config = SeedConfig::default();
        seed_config.max_seeds = 1;
        seed_config.seed_budget = 1;
        let n_params = rho.to_flat().len();
        let t_b = Instant::now();
        let result = OuterProblem::new(n_params)
            .with_seed_config(seed_config)
            .with_initial_rho(rho.to_flat())
            .run(&mut obj_full, &format!("profile p={p}"))
            .expect("outer engine must complete");
        let full_s = t_b.elapsed().as_secs_f64();
        obj_full
            .certify_outer_result(&result)
            .expect("profile outer result must certify the installed state");
        let fitted = obj_full
            .into_fitted()
            .expect("outer fit was evaluated")
            .term;
        let ev = global_ev(z.view(), fitted.fitted().view());

        println!(
            "[PROFILE p={p} M={M}] curvature_walk={walk_s:.2}s (arrived={arrived:?})  \
             full_outer_fit={full_s:.2}s  post_walk_est={:.2}s  EV={ev:.4}  \
             walk_fraction={:.0}%  ||  COLD_no_walk={cold_s:.2}s EV={cold_ev:.4} (ok={})",
            (full_s - walk_s).max(0.0),
            100.0 * walk_s / full_s.max(1.0e-9),
            cold.is_ok()
        );

        assert!(
            ev >= 0.90,
            "p={p}: profiling fit should still recover the circle, EV={ev:.4}"
        );
    }
}
