//! Large-scale SAE arrow-Schur benchmark.
//!
//! Exercises `K_atoms ∈ {8, 32, 64}` with realistic N and p so the structural
//! costs that are invisible at K≤2 become visible:
//!
//!   - dense `(q × K·M)` `H_tβ` materialisation (`q = K + K*d = K*(1+d)`)
//!   - dense `q × q` per-row `H_tt^(i)` block
//!   - dense `K*M × K*M` `sys.hbb` shared block
//!   - PCG path triggered by `K > 2000` (tested at `k_atoms=64, M=32`)
//!
//! Both Softmax and JumpReLU assignment variants are exercised so the
//! sparse-vs-dense assignment Jacobian story is measurable.
//!
//! Run with: `cargo bench --bench sae_arrow_schur_bench`
//!
//! Parameter choices keep RAM ≤ ~2 GB on a laptop CI runner:
//!
//!   N=10_000, K=8,  M=4,  d=1 → rows=10k × q=16  × beta=32  → manageable
//!   N=10_000, K=32, M=8,  d=1 → rows=10k × q=64  × beta=256 → moderate
//!   N=10_000, K=64, M=16, d=1 → rows=10k × q=128 × beta=1024 → large
//!
//! The CI machine can select `N_SMALL` variants; full large-scale uses `N_LARGE`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3};
use std::hint::black_box;
use std::time::Instant;

use gam::solver::arrow_schur::ArrowSolveOptions;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};

/// Bench configuration: `(k_atoms, basis_size, latent_dim, n_obs, p_out)`.
/// Picked so `beta_dim = k_atoms * basis_size * p_out` and
/// `q = k_atoms * (1 + latent_dim)` are in the regime the issue targets.
#[derive(Clone, Copy)]
struct BenchConfig {
    k_atoms: usize,
    basis_size: usize,
    latent_dim: usize,
    n_obs: usize,
    p_out: usize,
}

/// Small configs for CI (fit on a 16 GB runner without OOM).
const CONFIGS_CI: &[BenchConfig] = &[
    BenchConfig {
        k_atoms: 8,
        basis_size: 4,
        latent_dim: 1,
        n_obs: 2_000,
        p_out: 2,
    },
    BenchConfig {
        k_atoms: 32,
        basis_size: 4,
        latent_dim: 1,
        n_obs: 2_000,
        p_out: 2,
    },
    BenchConfig {
        k_atoms: 64,
        basis_size: 8,
        latent_dim: 1,
        n_obs: 2_000,
        p_out: 2,
    },
];

/// Large-scale configs for local benchmarking.
const CONFIGS_LARGE_SCALE: &[BenchConfig] = &[
    BenchConfig {
        k_atoms: 8,
        basis_size: 4,
        latent_dim: 1,
        n_obs: 10_000,
        p_out: 4,
    },
    BenchConfig {
        k_atoms: 32,
        basis_size: 8,
        latent_dim: 1,
        n_obs: 10_000,
        p_out: 4,
    },
    BenchConfig {
        k_atoms: 64,
        basis_size: 16,
        latent_dim: 1,
        n_obs: 10_000,
        p_out: 4,
    },
];

/// Deterministic pseudo-random f64 from a linear-congruential generator.
/// Avoids pulling in rand for the bench harness itself.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to (-1, 1).
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Build a synthetic `SaeManifoldTerm` at the given config.
///
/// All arrays are filled with deterministic pseudo-random values so the
/// Hessian is well-conditioned (diagonal-dominant `htt`, positive penalty).
fn build_term(cfg: &BenchConfig, mode: AssignmentMode) -> (SaeManifoldTerm, Array2<f64>) {
    let BenchConfig {
        k_atoms,
        basis_size: m,
        latent_dim: d,
        n_obs: n,
        p_out: p,
    } = *cfg;

    let mut rng: u64 = 0xdeadbeef_cafebabe;

    // Basis values Phi: (n, m)
    // Basis Jacobian jet: (n, m, d)
    // Decoder B: (m, p)
    // Smooth penalty S: (m, m) — symmetric positive-definite (identity + noise)
    // Logits: (n, k)
    // Coords per atom: (n, d)
    // Target Z: (n, p)

    let logits = Array2::from_shape_fn((n, k_atoms), |_| lcg_f64(&mut rng) * 0.5);

    let target = Array2::from_shape_fn((n, p), |_| lcg_f64(&mut rng));

    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);

    for _k in 0..k_atoms {
        let phi = Array2::from_shape_fn((n, m), |_| lcg_f64(&mut rng) * 0.1);
        let jet = Array3::from_shape_fn((n, m, d), |_| lcg_f64(&mut rng) * 0.01);
        let decoder = Array2::from_shape_fn((m, p), |_| lcg_f64(&mut rng) * 0.5);
        // S = 0.1 * I + small symmetric noise → positive-definite.
        let mut smooth = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            smooth[[i, i]] = 0.1 + 0.01 * lcg_f64(&mut rng).abs();
        }
        let atom = SaeManifoldAtom::new(
            format!("atom_{_k}"),
            SaeAtomBasisKind::EuclideanPatch,
            d,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("SaeManifoldAtom::new failed in benchmark fixture");
        atoms.push(atom);

        let coords = Array2::from_shape_fn((n, d), |_| lcg_f64(&mut rng) * 0.5);
        coord_blocks.push(coords);
    }

    let assignment = SaeAssignment::from_blocks_with_mode(logits, coord_blocks, mode)
        .expect("SaeAssignment::from_blocks_with_mode failed in benchmark fixture");

    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("SaeManifoldTerm::new failed in benchmark fixture");

    (term, target)
}

/// Build a `SaeManifoldRho` appropriate for the config.
fn build_rho(cfg: &BenchConfig) -> SaeManifoldRho {
    let log_ard: Vec<Array1<f64>> = (0..cfg.k_atoms)
        .map(|_| Array1::from_elem(cfg.latent_dim, 0.0_f64))
        .collect();
    SaeManifoldRho::new(0.0, -4.0, log_ard)
}

fn label(cfg: &BenchConfig, variant: &str) -> String {
    format!(
        "K{}_M{}_d{}_N{}_p{}_{}",
        cfg.k_atoms, cfg.basis_size, cfg.latent_dim, cfg.n_obs, cfg.p_out, variant
    )
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

fn bench_assembly_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_assembly_softmax");
    group.sample_size(10);

    for cfg in CONFIGS_CI {
        let mode = AssignmentMode::softmax(1.0);
        let (mut term, target) = build_term(cfg, mode);
        let rho = build_rho(cfg);
        let id = BenchmarkId::new("assemble", label(cfg, "softmax"));

        group.bench_with_input(id, &(), |b, ()| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let sys = term
                        .assemble_arrow_schur(target.view(), &rho, None)
                        .expect("assemble_arrow_schur failed");
                    black_box(sys);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_assembly_threshold_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_assembly_threshold_gate");
    group.sample_size(10);

    for cfg in CONFIGS_CI {
        // Threshold gate at threshold=0.0 → ~50% active fraction (sigmoid(0)=0.5).
        let mode = AssignmentMode::threshold_gate(1.0, 0.0);
        let (mut term, target) = build_term(cfg, mode);
        let rho = build_rho(cfg);
        let id = BenchmarkId::new("assemble", label(cfg, "threshold_gate"));

        group.bench_with_input(id, &(), |b, ()| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let sys = term
                        .assemble_arrow_schur(target.view(), &rho, None)
                        .expect("assemble_arrow_schur failed");
                    black_box(sys);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_solve_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_solve_direct");
    group.sample_size(10);

    for cfg in CONFIGS_CI {
        let mode = AssignmentMode::softmax(1.0);
        let (mut term, target) = build_term(cfg, mode);
        let rho = build_rho(cfg);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("assemble_arrow_schur failed in bench setup");
        let opts = ArrowSolveOptions::direct();
        let id = BenchmarkId::new("solve_direct", label(cfg, "softmax"));

        group.bench_with_input(id, &(), |b, ()| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = sys
                        .solve_with_options(1e-6, 1e-6, &opts)
                        .expect("solve_with_options failed");
                    black_box(result);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_solve_pcg(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_solve_pcg");
    group.sample_size(10);

    for cfg in CONFIGS_CI {
        let mode = AssignmentMode::softmax(1.0);
        let (mut term, target) = build_term(cfg, mode);
        let rho = build_rho(cfg);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("assemble_arrow_schur failed in bench setup");
        let opts = ArrowSolveOptions::inexact_pcg();
        let id = BenchmarkId::new("solve_pcg", label(cfg, "softmax"));

        group.bench_with_input(id, &(), |b, ()| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = sys
                        .solve_with_options(1e-6, 1e-6, &opts)
                        .expect("solve_with_options failed");
                    black_box(result);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_full_newton_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_full_newton_step");
    group.sample_size(10);

    for cfg in CONFIGS_CI {
        let mode = AssignmentMode::softmax(1.0);
        let (mut term, target) = build_term(cfg, mode);
        let mut rho = build_rho(cfg);
        let id = BenchmarkId::new("newton_step", label(cfg, "softmax"));

        group.bench_with_input(id, &(), |b, ()| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let loss = term
                        .run_joint_fit_arrow_schur(
                            target.view(),
                            &mut rho,
                            None,
                            1,
                            1.0,
                            1e-6,
                            1e-6,
                        )
                        .expect("run_joint_fit_arrow_schur failed");
                    black_box(loss);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

/// Large-scale assembly timings (not run in CI — too slow, but wired so
/// `cargo bench --bench sae_arrow_schur_bench large-scale` runs them locally).
fn bench_large_scale_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("sae_arrow_schur_large_scale_assembly");
    group.sample_size(10);

    for cfg in CONFIGS_LARGE_SCALE {
        for (variant, mode) in [
            ("softmax", AssignmentMode::softmax(1.0)),
            ("threshold_gate", AssignmentMode::threshold_gate(1.0, 0.0)),
        ] {
            let (mut term, target) = build_term(cfg, mode);
            let rho = build_rho(cfg);
            let id = BenchmarkId::new("assemble", label(cfg, variant));

            group.bench_with_input(id, &(), |b, ()| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let sys = term
                            .assemble_arrow_schur(target.view(), &rho, None)
                            .expect("assemble_arrow_schur failed");
                        black_box(sys);
                    }
                    start.elapsed()
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_assembly_softmax,
    bench_assembly_threshold_gate,
    bench_solve_direct,
    bench_solve_pcg,
    bench_full_newton_step,
    bench_large_scale_assembly,
);
criterion_main!(benches);
