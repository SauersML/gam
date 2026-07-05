//! Births/sec: serial [`fit_stagewise`] vs parallel [`fit_stagewise_batched`] on a
//! small axis-aligned dense-torus replica (the co-acceptance regime — circles in
//! orthogonal ambient planes are output-dim-disjoint, so the batched driver races
//! their K=1 fits across cores and co-accepts a disjoint batch per residual
//! snapshot instead of one birth per serial round).
//!
//! A plain `main` (harness = false), seconds-scale, no full-size runs: it prints
//! births, wall-time, births/sec for each driver, the speedup, and the batched
//! driver's max per-round co-acceptance (the concurrency multiplier). Run with
//! `cargo bench --bench stagewise_batched_births`.

use std::time::Instant;

use gam::terms::sae::manifold::{
    BatchedStagewiseConfig, StagewiseConfig, fit_stagewise, fit_stagewise_batched,
};
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn lcg_normal(state: &mut u64) -> f64 {
    let u1 = lcg_uniform(state).max(1e-12);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Axis-aligned dense torus: circle `c` in ambient plane `{2c, 2c+1}`, dense, with
/// independent uniform phase per circle — the certifiable, output-dim-disjoint
/// regime the batched driver co-accepts.
fn planted_axis_dense_circles(n: usize, p: usize, k: usize, sigma: f64, seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut data = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for c in 0..k {
            let th = std::f64::consts::TAU * lcg_uniform(&mut state);
            data[[i, 2 * c]] += th.cos();
            data[[i, 2 * c + 1]] += th.sin();
        }
        for j in 0..p {
            data[[i, j]] += sigma * lcg_normal(&mut state);
        }
    }
    data
}

/// An UNFIT K=1 softmax circle-atom seed on ambient dirs (0,1), active on every
/// row. Unfit + softmax on purpose (matching the driver parity test): a warm K=1
/// joint fit chases a rank-2 blend across all circles and contaminates the residual
/// so the κ certificate rejects it, and a ThresholdGate starves the born circle's
/// own-presence gate below the birth threshold; an unfit softmax seed leaves round 1
/// a clean multi-circle residual the drivers actually grow K on.
fn build_seed(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let atom = SaeManifoldAtom::new(
        "seed".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let n = coords.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    term.set_guards_enabled(false);
    (term, rho)
}

fn main() {
    let n = 900usize;
    let p = 16usize;
    let q = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);
    let target = planted_axis_dense_circles(n, p, q, 0.03, 0x2111_A11E_u64);

    let mut config = StagewiseConfig::default();
    config.inner_max_iter = 24;
    config.max_births = 8;
    config.max_backfit_sweeps = 1;
    config.min_effect_ev = 0.0;
    config.structured_whitening = false;

    let (seed_s, rho_s) = build_seed(&evaluator, &coords, p);
    let (seed_b, rho_b) = build_seed(&evaluator, &coords, p);

    let t0 = Instant::now();
    let serial = fit_stagewise(seed_s, rho_s, target.view(), None, None, &config, None, None)
        .expect("serial driver");
    let serial_dt = t0.elapsed().as_secs_f64();

    let batch_config = BatchedStagewiseConfig {
        base: config,
        max_candidates_per_round: 8,
    };
    let t1 = Instant::now();
    let batched = fit_stagewise_batched(seed_b, rho_b, target.view(), None, None, &batch_config)
        .expect("batched driver");
    let batched_dt = t1.elapsed().as_secs_f64();

    let sb = serial.report.births_accepted;
    let bb = batched.report.births_accepted;
    let max_co_accept = batched
        .batch_records
        .iter()
        .map(|r| r.co_accepted)
        .max()
        .unwrap_or(0);
    let serial_rounds = serial.report.birth_records.len();
    let batched_rounds = batched.batch_records.len();

    println!("cores available (rayon): {}", rayon::current_num_threads());
    println!(
        "serial : births={sb} rounds={serial_rounds} time={serial_dt:.4}s births/s={:.2}",
        sb as f64 / serial_dt.max(1e-9)
    );
    println!(
        "batched: births={bb} rounds={batched_rounds} time={batched_dt:.4}s births/s={:.2} max_co_accept={max_co_accept}",
        bb as f64 / batched_dt.max(1e-9)
    );
    println!(
        "speedup (serial_time / batched_time): {:.2}x   births/round: serial={:.2} batched={:.2}",
        serial_dt / batched_dt.max(1e-9),
        sb as f64 / serial_rounds.max(1) as f64,
        bb as f64 / batched_rounds.max(1) as f64,
    );
}
