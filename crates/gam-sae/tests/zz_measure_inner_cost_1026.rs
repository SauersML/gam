//! #1026 / #2230 — zz_measure diagnostic: per-call wall cost of the inner
//! joint fit at the BSF-zoo MICRO shape (the shape whose MSI fit burned 6+ h
//! in outer-criterion churn: 12 atoms, 3000 rows, 48-dim ambient).
//!
//! The MSI zoo log shows one `run_joint_fit_arrow_schur`-terminated criterion
//! evaluation every 70–220 s. This measure separates the two candidate walls:
//!
//!  * if ONE inner driver call at this shape costs tens of seconds locally,
//!    the wall is the inner solve itself (per-iteration assembly/factor cost)
//!    and the basin-envelope fix only shrinks the multiplier;
//!  * if it costs well under a second, the wall is purely the outer
//!    EVALUATION MULTIPLICITY (hysteretic criterion churn, #2230), and the
//!    basin-bundle envelope is the whole fix.
//!
//! Signal test (zz_measure discipline): eprintln the timings; the only hard
//! asserts are finiteness + descent, so the measure never becomes a flaky
//! wall-clock gate.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, sae_pca_seed_initial_coords,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, s};
use std::sync::Arc;
use std::time::Instant;

/// Deterministic standard-normal stream (splitmix64 + Box–Muller), as in
/// `sae_ev_vs_k_1026.rs` — identical run to run and across thread counts.
fn normal_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut cached: Option<f64> = None;
    move || {
        if let Some(v) = cached.take() {
            return v;
        }
        let u1 = ((next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let (s_, c) = (std::f64::consts::TAU * u2).sin_cos();
        cached = Some(r * s_);
        r * c
    }
}

/// Zoo-micro-like target: a superposition of a few planted circles + linear
/// segments in a 48-dim ambient with small noise. Matches the m12 job's
/// (n=3000, p=48) scale; the exact factor kinds are irrelevant to the wall
/// being measured (assembly/factorization cost is shape-driven).
fn planted_micro(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut normal = normal_stream(seed);
    // 4 planted circle factors on random 2-frames + noise.
    let frames: Vec<(Array1<f64>, Array1<f64>)> = (0..4)
        .map(|_| {
            let u = Array1::from_shape_fn(p, |_| normal());
            let v = Array1::from_shape_fn(p, |_| normal());
            (u, v)
        })
        .collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for (f, (u, v)) in frames.iter().enumerate() {
            let theta = std::f64::consts::TAU * ((i * (f + 3)) as f64) / (n as f64);
            for c in 0..p {
                z[[i, c]] += theta.cos() * u[c] + theta.sin() * v[c];
            }
        }
        for c in 0..p {
            z[[i, c]] += 0.05 * normal();
        }
    }
    z
}

#[test]
fn zz_measure_inner_joint_fit_cost_at_zoo_micro_shape() {
    let (n, p, k_atoms, num_basis) = (3000usize, 48usize, 12usize, 8usize);
    let z = planted_micro(n, p, 7);

    // Production PCA seed for all K charts (the cold-path seeder), then
    // per-atom ridge-LSQ decoders — the same construction the K=1 harness in
    // `sae_ev_vs_k_1026.rs` uses, replicated per atom.
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k_atoms];
    let atom_dims = vec![1usize; k_atoms];
    let t_seed = Instant::now();
    let seed_coords =
        sae_pca_seed_initial_coords(z.view(), &basis_kinds, &atom_dims).expect("pca seed");
    let seed_secs = t_seed.elapsed().as_secs_f64();

    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for a in 0..k_atoms {
        let coords = seed_coords.slice(s![a, .., 0..1]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let mut xtx = fast_atb(&phi, &phi);
        for i in 0..m {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z);
        let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz) / (k_atoms as f64);
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("circle_{a}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(evaluator.clone()),
        );
        coord_blocks.push(coords);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k_atoms)),
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let mut rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![Array1::from_elem(1, 1.0e-3_f64.ln()); k_atoms],
    );

    let loss0 = term.loss(z.view(), &rho).unwrap().total();

    // One CHUNK of the inner driver at the historical probe chunk width
    // (inner_max_iter = 12, the zoo m12 setting), timed end to end.
    let t_fit = Instant::now();
    let loss = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 12, 1.0, 1.0e-6, 1.0e-6)
        .expect("inner joint fit runs at the zoo micro shape");
    let fit_secs = t_fit.elapsed().as_secs_f64();

    // A second warm call (the envelope's member re-convergence cost model:
    // "members near their optimum re-converge in a round or two").
    let t_warm = Instant::now();
    let loss_warm = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 12, 1.0, 1.0e-6, 1.0e-6)
        .expect("warm inner joint fit runs");
    let warm_secs = t_warm.elapsed().as_secs_f64();

    eprintln!(
        "[zz_measure #1026 inner-cost] shape=(n={n}, p={p}, K={k_atoms}, m={num_basis}) \
         pca_seed={seed_secs:.3}s cold_12it={fit_secs:.3}s warm_12it={warm_secs:.3}s \
         loss0={loss0:.6} loss_cold={:.6} loss_warm={:.6}",
        loss.total(),
        loss_warm.total()
    );

    assert!(loss.total().is_finite() && loss_warm.total().is_finite());
    assert!(
        loss.total() <= loss0 + 1.0e-8,
        "inner driver must not increase the penalized loss: {loss0} -> {}",
        loss.total()
    );
}
