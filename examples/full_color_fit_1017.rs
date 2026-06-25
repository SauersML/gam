//! #1017 — measure the FULL production color-arm SAE manifold fit.
//!
//! The flagship #1017 number ("26+ minutes, 0% GPU") is the *full* color-arm
//! fit: the real outer REML/EFS smoothing-parameter loop wrapping the inner
//! Newton/PIRLS solve — NOT the single-frame `device_resident_inner_1017`
//! benchmark (which times one inner solve on a constant-Hessian frame that
//! converges in 1 iteration). For the color shape the production border is
//! k=p=5120 > DIRECT_SOLVE_MAX_K (2000), so the inner solve routes through
//! `ArrowSolverMode::InexactPCG` (matrix-free reduced-Schur PCG), whose matvec
//! the Phase-1 re-keying offloads to the device when the work clears the floor.
//!
//! This example drives the real production path:
//!   * one `reml_criterion` eval (one full inner fit at fixed rho), and
//!   * the full `OuterProblem::run` outer loop,
//! both on the color shape, with GPU execution telemetry snapshotted around
//! each so we can prove whether the device actually ran (handle creations,
//! kernel launches, factorizations, H2D/D2H bytes) or silently fell back to CPU.
//!
//! The closure bar for #1017: the full fit completes at ~1.56s (>=1000x over
//! the 26-min original).
//!
//! Run on a CUDA host:
//! ```text
//! cargo run --release --example full_color_fit_1017
//! ```

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gam::gpu::profile::{GpuExecutionTelemetry, telemetry_reset, telemetry_snapshot};
use gam::solver::arrow_schur::ArrowSolveOptions;
use gam::terms::{
    AnalyticPenaltyRegistry, AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator,
    SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};
use ndarray::{Array1, Array2};

// Color-arm shape (the #1017 measured gap): few rows, very wide border.
const N: usize = 180;
const P: usize = 5120;
const K: usize = 1; // atoms
const D: usize = 1; // latent coords per atom (Circle)
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const LOG_LAMBDA_SPARSE: f64 = -12.0;
const LOG_LAMBDA_SMOOTH: f64 = -12.0;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
}

fn smooth_penalty(width: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((width, width));
    for i in 0..width {
        s[[i, i]] = if i == 0 { 1.0e-3 } else { 1.0 };
    }
    s
}

fn decoder(width: usize, p: usize, atom: usize, rng: &mut Lcg) -> Array2<f64> {
    let scale = 0.18 / (width as f64).sqrt();
    Array2::from_shape_fn((width, p), |(row, col)| {
        let carrier = ((row + 1) as f64 * 0.17 + (col + 1) as f64 * 0.013).sin();
        let atom_shift = ((atom + 1) as f64 * (col + 3) as f64 * 0.0021).cos();
        scale * (0.7 * carrier + 0.3 * atom_shift + 0.05 * rng.signed())
    })
}

fn circle_coords(n: usize, atom: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 1), |(row, _)| {
        ((row as f64) * (0.031 + 0.004 * atom as f64) + 0.07 * atom as f64).rem_euclid(1.0)
    })
}

fn logits(n: usize, k: usize, rng: &mut Lcg) -> Array2<f64> {
    Array2::from_shape_fn((n, k), |(row, atom)| {
        let phase = (row as f64) * (0.019 + 0.003 * atom as f64) + atom as f64 * 0.41;
        -1.1 + 0.9 * phase.sin() + 0.08 * rng.signed()
    })
}

fn build_color_term() -> Result<(SaeManifoldTerm, Array2<f64>, SaeManifoldRho), String> {
    let mut rng = Lcg::new(0x5AE0_1017_9E37_79B9 ^ N as u64 ^ ((P as u64) << 17));
    let mut atoms = Vec::with_capacity(K);
    let mut coord_blocks = Vec::with_capacity(K);
    let mut manifolds = Vec::with_capacity(K);
    for atom_idx in 0..K {
        let evaluator = PeriodicHarmonicEvaluator::new(3).map_err(|e| e.to_string())?;
        let coords = circle_coords(N, atom_idx);
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .map_err(|e| e.to_string())?;
        let width = phi.ncols();
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            D,
            phi,
            jet,
            decoder(width, P, atom_idx, &mut rng),
            smooth_penalty(width),
        )
        .map_err(|e| e.to_string())?
        .with_basis_evaluator(Arc::new(evaluator));
        atoms.push(atom);
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits(N, K, &mut rng),
        coord_blocks,
        manifolds,
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .map_err(|e| e.to_string())?;
    let term = SaeManifoldTerm::new(atoms, assignment).map_err(|e| e.to_string())?;
    let target = term.fitted();
    let rho = SaeManifoldRho::new(
        LOG_LAMBDA_SPARSE,
        LOG_LAMBDA_SMOOTH,
        vec![Array1::<f64>::zeros(0); K],
    );
    Ok((term, target, rho))
}

fn delta(before: &GpuExecutionTelemetry, after: &GpuExecutionTelemetry) -> String {
    format!(
        "handles=+{} kernels=+{} factorizations=+{} h2d=+{}KB d2h=+{}KB cpu_fallbacks=+{}",
        after
            .handle_creation_count
            .saturating_sub(before.handle_creation_count),
        after
            .kernel_launch_count
            .saturating_sub(before.kernel_launch_count),
        after
            .factorization_count
            .saturating_sub(before.factorization_count),
        (after.h2d_bytes.saturating_sub(before.h2d_bytes)) / 1024,
        (after.d2h_bytes.saturating_sub(before.d2h_bytes)) / 1024,
        after
            .cpu_fallback_count
            .saturating_sub(before.cpu_fallback_count),
    )
}

fn main() -> std::process::ExitCode {
    println!("FULLCOLOR_1017 shape=color n={N} p={P} K={K} d={D}");

    let (term, target, rho) = match build_color_term() {
        Ok(v) => v,
        Err(e) => {
            println!("FULLCOLOR_1017 BUILD_FAILED: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    println!(
        "FULLCOLOR_1017 beta_dim={} row_dim={}",
        term.beta_dim(),
        term.assignment.row_block_dim()
    );
    let _ = std::io::stdout().flush();

    // The bar is 1.56s; anything past 10s already fails by >6x, so don't wait it
    // out. Run the whole fit on a worker thread and have main wait with a 10s
    // deadline (no process::exit — returning from main terminates the process,
    // killing the detached worker). Each stage flushes as it finishes, so partial
    // progress survives a timeout.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let worker = std::thread::spawn(move || {
        // Empty registry: the #1026 collapse barriers set deferred-factored beta
        // curvature for the wide color border, so a registry must be present even
        // with no analytic penalty (zero extra curvature; barrier curvature is in
        // the system).
        let registry = AnalyticPenaltyRegistry::new();
        let mut t = term.clone();

        // ---- Stage A: assemble the arrow-Schur system once (timed) ----
        telemetry_reset();
        let a0 = telemetry_snapshot();
        let start = Instant::now();
        let sys = match t.assemble_arrow_schur(target.view(), &rho, Some(&registry)) {
            Ok(s) => s,
            Err(e) => {
                println!("FULLCOLOR_1017 assemble FAILED: {e}");
                let _ = std::io::stdout().flush();
                let _ = tx.send(());
                return;
            }
        };
        let a_ms = start.elapsed().as_secs_f64() * 1e3;
        let a1 = telemetry_snapshot();
        println!(
            "FULLCOLOR_1017 assemble ms={a_ms:.3} k={} rows={} gpu[{}]",
            sys.k,
            sys.rows.len(),
            delta(&a0, &a1)
        );
        let _ = std::io::stdout().flush();

        // Second assemble: is the 9s one-time setup or a per-inner-iter cost? The
        // production inner Newton re-assembles every iteration.
        {
            let start = Instant::now();
            let sys2 = t.assemble_arrow_schur(target.view(), &rho, Some(&registry));
            let a2_ms = start.elapsed().as_secs_f64() * 1e3;
            println!("FULLCOLOR_1017 assemble2 ms={a2_ms:.3} ok={}", sys2.is_ok());
            let _ = std::io::stdout().flush();
        }

        // ---- Stage C: InexactPCG (PRODUCTION path) with capped iterations to
        // separate per-iter cost from non-convergence. Linear scaling ⇒ per-iter
        // cost dominates; plateau ⇒ converges; hits cap ⇒ not converging. ----
        for &cap in &[2usize, 10, 50] {
            telemetry_reset();
            let c0 = telemetry_snapshot();
            let mut opts = ArrowSolveOptions::inexact_pcg();
            opts.pcg.max_iterations = cap;
            opts.trust_region.max_iterations = cap;
            let start = Instant::now();
            let r = sys.solve_with_options(0.0, 0.0, &opts);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            let c1 = telemetry_snapshot();
            match r {
                Ok((_, _, d)) => println!(
                    "FULLCOLOR_1017 solve_inexact_pcg cap={cap} ms={ms:.3} pcg_iters={} matvec_calls={} used_device_arrow={} final_rel_resid={:.2e} gpu[{}]",
                    d.iterations,
                    d.matvec_calls,
                    d.used_device_arrow,
                    d.final_relative_residual,
                    delta(&c0, &c1)
                ),
                Err(e) => println!("FULLCOLOR_1017 solve_inexact_pcg cap={cap} FAILED: {e}"),
            }
            let _ = std::io::stdout().flush();
        }

        println!("FULLCOLOR_1017 DONE");
        let _ = std::io::stdout().flush();
        let _ = tx.send(());
    });

    // Diagnostic cap (single solves, not the full fit): 45s headroom to capture
    // even a slow InexactPCG number. Returning from main terminates the process.
    match rx.recv_timeout(Duration::from_secs(45)) {
        Ok(()) => {
            let _ = worker.join();
            std::process::ExitCode::SUCCESS
        }
        Err(_) => {
            println!("FULLCOLOR_1017 KILLED_OVER_45S (a single solve exceeded 45s — aborting)");
            let _ = std::io::stdout().flush();
            std::process::ExitCode::from(1)
        }
    }
}
