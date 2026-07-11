//! #1017 — run the complete production color-arm SAE REML fit.
//!
//! This executable is a convergence and execution-path probe, not a deadline
//! benchmark. It runs the real `SaeManifoldOuterObjective` through the shared
//! `OuterProblem`, refuses to materialize a fitted model unless the outer
//! optimizer reports convergence, and only then reports elapsed time and GPU
//! telemetry. There is no wall-clock deadline, iteration-cap sweep, or grid
//! search; work interrupted by a scheduler wall is preserved by the production
//! SAE checkpoint/resume path owned by the objective.
//!
//! The color arm has `N=180`, ambient output width `P=5120`, and a three-column
//! periodic basis. Its full decoder therefore has `beta_dim=3*5120=15360`.
//! Production may profile that decoder through its exact low-rank Grassmann
//! frame before assembling the Newton system, so the executable separately
//! reports the post-fit `factored_border_dim`, assembled border width, and the
//! solver mode selected from that actual border. Conflating `beta_dim` with the
//! factored border was the source of the old executable's stale InexactPCG
//! claim.
//!
//! Run on a CUDA host:
//! ```text
//! cargo run --release --example full_color_fit_1017
//! ```

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gam::gpu::profile::{GpuExecutionTelemetry, telemetry_reset, telemetry_snapshot};
use gam::solver::arrow_schur::ArrowSolverMode;
use gam::solver::rho_optimizer::OuterProblem;
use gam::solver::seeding::SeedConfig;
use gam::terms::{
    AnalyticPenaltyRegistry, sae::manifold::AssignmentMode, latent::LatentManifold, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldOuterObjective,
    sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};
use gam_sae::manifold::StagewiseConfig;
use ndarray::{Array1, Array2};

// Production color-arm shape from #1017: few rows, very wide ambient output.
const N: usize = 180;
const P: usize = 5120;
const N_ATOMS: usize = 1;
const LATENT_DIM: usize = 1;
const PERIODIC_BASIS_WIDTH: usize = 3;

const LOG_LAMBDA_SPARSE: f64 = -12.0;
const LOG_LAMBDA_SMOOTH: f64 = -12.0;
const GATE_TEMPERATURE: f64 = 0.5;
const IBP_CONCENTRATION: f64 = 1.0;

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

/// Exact final-function roughness Gram for the periodic Fourier basis.
///
/// For `f(t)=B[0,:]+sum_h(B[sin_h,:] sin(2pi h t)+B[cos_h,:] cos(2pi h t))`,
/// `integral_0^1 ||f''(t)||^2 dt = trace(B^T S B)` with diagonal entries
/// `S_sin_h=S_cos_h=(2pi h)^4/2` and the constant in the exact null space.
/// Thus the coefficient quadratic consumed by the solver is precisely the
/// final-function penalty, rather than an arbitrary coefficient shrinkage.
fn periodic_second_derivative_roughness(width: usize) -> Result<Array2<f64>, String> {
    if width == 0 || width % 2 == 0 {
        return Err(format!(
            "periodic Fourier roughness requires an odd positive width; got {width}"
        ));
    }
    let mut penalty = Array2::<f64>::zeros((width, width));
    for harmonic in 1..=(width - 1) / 2 {
        let omega = std::f64::consts::TAU * harmonic as f64;
        let integrated_second_derivative_energy = omega.powi(4) / 2.0;
        penalty[[2 * harmonic - 1, 2 * harmonic - 1]] = integrated_second_derivative_energy;
        penalty[[2 * harmonic, 2 * harmonic]] = integrated_second_derivative_energy;
    }
    Ok(penalty)
}

fn decoder(width: usize, p: usize, rng: &mut Lcg) -> Array2<f64> {
    let scale = 0.18 / (width as f64).sqrt();
    Array2::from_shape_fn((width, p), |(row, col)| {
        let carrier = ((row + 1) as f64 * 0.17 + (col + 1) as f64 * 0.013).sin();
        let shift = ((col + 3) as f64 * 0.0021).cos();
        scale * (0.7 * carrier + 0.3 * shift + 0.05 * rng.signed())
    })
}

fn circle_coords(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, LATENT_DIM), |(row, _)| {
        ((row as f64) * 0.031).rem_euclid(1.0)
    })
}

fn logits(n: usize, rng: &mut Lcg) -> Array2<f64> {
    Array2::from_shape_fn((n, N_ATOMS), |(row, _)| {
        -1.1 + 0.9 * ((row as f64) * 0.019).sin() + 0.08 * rng.signed()
    })
}

fn build_color_term() -> Result<(SaeManifoldTerm, Array2<f64>, SaeManifoldRho), String> {
    let mut rng = Lcg::new(0x5AE0_1017_9E37_79B9 ^ N as u64 ^ ((P as u64) << 17));
    let evaluator = PeriodicHarmonicEvaluator::new(PERIODIC_BASIS_WIDTH)?;
    let coords = circle_coords(N);
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let width = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "color_circle",
        SaeAtomBasisKind::Periodic,
        LATENT_DIM,
        phi,
        jet,
        decoder(width, P, &mut rng),
        periodic_second_derivative_roughness(width)?,
    )?
    .with_basis_evaluator(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits(N, &mut rng),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        // Match the #1017 production color-arm gate exactly; this explicit
        // diagnostic fixture does not alter the library's default assignment.
        AssignmentMode::ibp_map(GATE_TEMPERATURE, IBP_CONCENTRATION, false),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let target = term.fitted();
    let rho = SaeManifoldRho::new(
        LOG_LAMBDA_SPARSE,
        LOG_LAMBDA_SMOOTH,
        vec![Array1::<f64>::zeros(0); N_ATOMS],
    );
    Ok((term, target, rho))
}

fn telemetry_delta(before: &GpuExecutionTelemetry, after: &GpuExecutionTelemetry) -> String {
    format!(
        "handles=+{} kernels=+{} factorizations=+{} h2d=+{}KiB d2h=+{}KiB cpu_fallbacks=+{}",
        after
            .handle_creation_count
            .saturating_sub(before.handle_creation_count),
        after
            .kernel_launch_count
            .saturating_sub(before.kernel_launch_count),
        after
            .factorization_count
            .saturating_sub(before.factorization_count),
        after.h2d_bytes.saturating_sub(before.h2d_bytes) / 1024,
        after.d2h_bytes.saturating_sub(before.d2h_bytes) / 1024,
        after
            .cpu_fallback_count
            .saturating_sub(before.cpu_fallback_count),
    )
}

fn run() -> Result<(), String> {
    let (term, target, initial_rho) = build_color_term()?;
    let initial_beta_dim = term.beta_dim();
    if initial_beta_dim != PERIODIC_BASIS_WIDTH * P {
        return Err(format!(
            "color fixture beta_dim {initial_beta_dim} != {}*{P}",
            PERIODIC_BASIS_WIDTH
        ));
    }

    let registry = AnalyticPenaltyRegistry::new();
    // Single source of truth for production inner-fit controls. `inner_max_iter`
    // is a refinement chunk: the REML evaluator extends until its stationarity
    // certificate is reached and returns an error otherwise.
    let inner = StagewiseConfig::default();
    let initial_rho_flat = initial_rho.to_flat();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        target.clone(),
        Some(registry.clone()),
        initial_rho,
        inner.inner_max_iter,
        inner.learning_rate,
        inner.ridge_ext_coord,
        inner.ridge_beta,
    );
    // A single explicit initial state: the optimizer moves continuously in rho;
    // no seed lattice or grid is evaluated.
    let rho_dim = initial_rho_flat.len();
    let search_initial_rho = objective
        .try_resume_from_checkpoint(rho_dim)
        .map(Array1::from)
        .unwrap_or(initial_rho_flat);
    let problem = OuterProblem::new(rho_dim)
        .with_initial_rho(search_initial_rho)
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..SeedConfig::default()
        });

    telemetry_reset();
    let telemetry_before = telemetry_snapshot();
    let started = Instant::now();
    let outer = problem
        .run(&mut objective, "#1017 production color SAE")
        .map_err(|err| format!("full color REML fit failed: {err}"))?;
    if !outer.converged || outer.converged_via.is_none() {
        return Err(format!(
            "outer optimizer returned without a convergence certificate: \
             converged={} via={:?}",
            outer.converged, outer.converged_via
        ));
    }
    objective
        .certify_outer_result(&outer)
        .map_err(|err| format!("full color outer certificate rejected: {err}"))?;

    // Only consume/mint the fitted model after the convergence certificate.
    // A successful fit no longer needs its wall-survival checkpoint; match the
    // production front door's completion transaction before consuming state.
    objective.remove_checkpoint();
    let fitted = objective
        .into_fitted()
        .map_err(|err| format!("full color fit finalization failed: {err}"))?;
    let elapsed = started.elapsed();
    let telemetry_after = telemetry_snapshot();
    let mut fitted_term = fitted.term;
    let factored_border_dim = fitted_term.factored_border_dim();
    let assembled =
        fitted_term.assemble_arrow_schur(target.view(), &fitted.rho, Some(&registry))?;
    let actual_mode = ArrowSolverMode::automatic(assembled.k);

    println!(
        "FULLCOLOR_1017 converged=true via={:?} plan={:?} outer_iterations={} \
         criterion={:.12e} loss_total={:.12e}",
        outer.converged_via,
        outer.plan_used,
        outer.iterations,
        outer.final_value,
        fitted.loss.total(),
    );
    println!(
        "FULLCOLOR_1017 shape n={N} p={P} atoms={N_ATOMS} latent_dim={LATENT_DIM} \
         beta_dim={initial_beta_dim} factored_border_dim={factored_border_dim} \
         assembled_border_dim={} solver_mode={actual_mode:?}",
        assembled.k,
    );
    println!(
        "FULLCOLOR_1017 elapsed_seconds={:.6} gpu[{}]",
        elapsed.as_secs_f64(),
        telemetry_delta(&telemetry_before, &telemetry_after),
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("FULLCOLOR_1017 FAILED: {err}");
            ExitCode::FAILURE
        }
    }
}
