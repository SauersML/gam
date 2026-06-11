use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gam::gpu::sae_resident::{DeviceResidentArrowError, qwen_non_gating_fixture};
use gam::solver::arrow_schur::ArrowSolveOptions;
use gam::solver::estimate::EstimationError;
use gam::solver::outer_strategy::{
    EfsEval, OuterCapability, OuterEval, OuterObjective, OuterProblem, SeedOutcome,
};
use gam::terms::sae_manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};

const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const LOG_LAMBDA_SPARSE: f64 = 0.0;
const LOG_LAMBDA_SMOOTH: f64 = -4.0;
const INNER_MAX_ITER: usize = 3;
const OUTER_MAX_ITER: usize = 4;
const LEARNING_RATE: f64 = 0.6;
const RIDGE_EXT_COORD: f64 = 1.0e-4;
const RIDGE_BETA: f64 = 1.0e-4;
const DEVICE_RIDGE_T: f64 = 0.0;
const DEVICE_RIDGE_BETA: f64 = 0.0;
const DEVICE_PARITY_TOL: f64 = 1.0e-10;

#[derive(Clone, Copy)]
enum Topology {
    Circle,
    Euclidean,
}

struct Shape {
    name: &'static str,
    n: usize,
    p: usize,
    k: usize,
    d: usize,
    topology: Topology,
}

struct Fixture {
    shape: Shape,
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

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

struct CountingObjective {
    inner: SaeManifoldOuterObjective,
    eval_cost_count: usize,
    eval_count: usize,
    eval_efs_count: usize,
    seed_count: usize,
}

impl CountingObjective {
    fn new(inner: SaeManifoldOuterObjective) -> Self {
        Self {
            inner,
            eval_cost_count: 0,
            eval_count: 0,
            eval_efs_count: 0,
            seed_count: 0,
        }
    }
}

impl OuterObjective for CountingObjective {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.eval_cost_count += 1;
        self.inner.eval_cost(rho)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        self.eval_count += 1;
        self.inner.eval(rho)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        self.eval_efs_count += 1;
        self.inner.eval_efs(rho)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        self.seed_count += 1;
        self.inner.seed_inner_state(beta)
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.inner.allow_continuation_prewarm()
    }

    fn requires_continuation_path_entry(&self) -> bool {
        self.inner.requires_continuation_path_entry()
    }
}

fn shape_named(name: &str) -> Option<Shape> {
    match name {
        "tiny" => Some(Shape {
            name: "tiny",
            n: 300,
            p: 8,
            k: 1,
            d: 1,
            topology: Topology::Circle,
        }),
        "color" => Some(Shape {
            name: "color",
            n: 180,
            p: 5120,
            k: 1,
            d: 1,
            topology: Topology::Circle,
        }),
        "qwen" => Some(Shape {
            name: "qwen",
            n: 2000,
            p: 2048,
            k: 8,
            d: 2,
            topology: Topology::Euclidean,
        }),
        _ => None,
    }
}

fn ms(start: Instant) -> f64 {
    1000.0 * start.elapsed().as_secs_f64()
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

fn euclidean_coords(n: usize, d: usize, atom: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, d), |(row, axis)| {
        let a = (row as f64 + 1.0) * (0.009 + 0.001 * (axis + 1) as f64);
        let b = (atom as f64 + 1.0) * (0.13 + 0.03 * axis as f64);
        0.8 * (a + b).sin()
    })
}

fn logits(n: usize, k: usize, rng: &mut Lcg) -> Array2<f64> {
    Array2::from_shape_fn((n, k), |(row, atom)| {
        let phase = (row as f64) * (0.019 + 0.003 * atom as f64) + atom as f64 * 0.41;
        -1.1 + 0.9 * phase.sin() + 0.08 * rng.signed()
    })
}

fn build_fixture(shape: Shape) -> Result<Fixture, String> {
    let mut rng = Lcg::new(0x5AE0_1017_9E37_79B9 ^ shape.n as u64 ^ ((shape.p as u64) << 17));
    let mut atoms = Vec::with_capacity(shape.k);
    let mut coord_blocks = Vec::with_capacity(shape.k);
    let mut manifolds = Vec::with_capacity(shape.k);

    for atom_idx in 0..shape.k {
        match shape.topology {
            Topology::Circle => {
                let evaluator = PeriodicHarmonicEvaluator::new(3)?;
                let coords = circle_coords(shape.n, atom_idx);
                let (phi, jet) = evaluator.evaluate(coords.view())?;
                let width = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    format!("circle_{atom_idx}"),
                    SaeAtomBasisKind::Periodic,
                    shape.d,
                    phi,
                    jet,
                    decoder(width, shape.p, atom_idx, &mut rng),
                    smooth_penalty(width),
                )?
                .with_basis_evaluator(Arc::new(evaluator));
                atoms.push(atom);
                coord_blocks.push(coords);
                manifolds.push(LatentManifold::Circle { period: 1.0 });
            }
            Topology::Euclidean => {
                let evaluator = EuclideanPatchEvaluator::new(shape.d, 2)?;
                let coords = euclidean_coords(shape.n, shape.d, atom_idx);
                let (phi, jet) = evaluator.evaluate(coords.view())?;
                let width = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    format!("euclidean_{atom_idx}"),
                    SaeAtomBasisKind::EuclideanPatch,
                    shape.d,
                    phi,
                    jet,
                    decoder(width, shape.p, atom_idx, &mut rng),
                    smooth_penalty(width),
                )?
                .with_basis_evaluator(Arc::new(evaluator));
                atoms.push(atom);
                coord_blocks.push(coords);
                manifolds.push(LatentManifold::Euclidean);
            }
        }
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits(shape.n, shape.k, &mut rng),
        coord_blocks,
        manifolds,
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )?;
    let term = SaeManifoldTerm::new(atoms, assignment)?;
    let mut target = term.fitted();
    for row in 0..shape.n {
        for col in 0..shape.p {
            target[[row, col]] += 0.01 * rng.signed();
        }
    }
    let rho = SaeManifoldRho::new(
        LOG_LAMBDA_SPARSE,
        LOG_LAMBDA_SMOOTH,
        vec![Array1::<f64>::zeros(shape.d); shape.k],
    );
    Ok(Fixture {
        shape,
        term,
        target,
        rho,
    })
}

fn print_stage(shape: &Shape, stage: &str, elapsed_ms: f64, extra: &str) {
    if extra.is_empty() {
        println!(
            "PERF shape={} stage={} ms={:.3} n={} p={} K={} d={}",
            shape.name, stage, elapsed_ms, shape.n, shape.p, shape.k, shape.d
        );
    } else {
        println!(
            "PERF shape={} stage={} ms={:.3} n={} p={} K={} d={} {}",
            shape.name, stage, elapsed_ms, shape.n, shape.p, shape.k, shape.d, extra
        );
    }
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn run_device_inner_iter(shape: &Shape) -> Result<(), String> {
    let build_start = Instant::now();
    let workspace = qwen_non_gating_fixture().map_err(|err| err.to_string())?;
    let build_ms = ms(build_start);

    let cpu_start = Instant::now();
    let cpu = workspace
        .cpu_reference_step(DEVICE_RIDGE_T, DEVICE_RIDGE_BETA)
        .map_err(|err| format!("device_inner_iter CPU reference failed: {err}"))?;
    let cpu_ms = ms(cpu_start);

    let device_start = Instant::now();
    let device = match workspace.one_inner_iteration(DEVICE_RIDGE_T, DEVICE_RIDGE_BETA) {
        Ok(step) => step,
        Err(DeviceResidentArrowError::Unavailable { reason }) => {
            print_stage(
                shape,
                "device_inner_iter",
                ms(device_start),
                &format!(
                    "status=skipped reason=\"{}\" build_ms={build_ms:.3} cpu_ms={cpu_ms:.3} host_bytes={} device_bytes={}",
                    reason.replace('"', "'"),
                    workspace.host_shadow_bytes(),
                    workspace.resident_device_bytes()
                ),
            );
            return Ok(());
        }
        Err(err) => return Err(format!("device_inner_iter device solve failed: {err}")),
    };
    let device_ms = ms(device_start);

    let dt_err = max_abs_diff(
        cpu.delta_t.as_slice().ok_or("CPU delta_t not contiguous")?,
        device
            .delta_t
            .as_slice()
            .ok_or("device delta_t not contiguous")?,
    );
    let db_err = max_abs_diff(
        cpu.delta_beta
            .as_slice()
            .ok_or("CPU delta_beta not contiguous")?,
        device
            .delta_beta
            .as_slice()
            .ok_or("device delta_beta not contiguous")?,
    );
    let logdet_err = (cpu.log_det_hessian - device.log_det_hessian).abs();
    let max_err = dt_err.max(db_err).max(logdet_err);
    print_stage(
        shape,
        "device_inner_iter",
        device_ms,
        &format!(
            "status=ok build_ms={build_ms:.3} cpu_ms={cpu_ms:.3} speedup={:.3} max_abs_step_err={max_err:.3e} objective={:.6e} grad_norm={:.6e} used_device={} host_bytes={} device_bytes={}",
            cpu_ms / device_ms.max(f64::MIN_POSITIVE),
            device.objective,
            device.gradient_norm,
            device.used_device,
            workspace.host_shadow_bytes(),
            workspace.resident_device_bytes()
        ),
    );
    if max_err > DEVICE_PARITY_TOL {
        return Err(format!(
            "device_inner_iter parity failed: max_abs_step_err={max_err:e} > {DEVICE_PARITY_TOL:e}"
        ));
    }
    Ok(())
}

fn run(shape: Shape) -> Result<(), String> {
    let total_start = Instant::now();
    let fixture = build_fixture(shape)?;
    let beta_dim = fixture.term.beta_dim();
    let row_dim = fixture.term.assignment.row_block_dim();

    let mut term_for_assemble = fixture.term.clone();
    let start = Instant::now();
    let assembled =
        term_for_assemble.assemble_arrow_schur(fixture.target.view(), &fixture.rho, None)?;
    print_stage(
        &fixture.shape,
        "assemble_arrow_schur",
        ms(start),
        &format!(
            "beta_dim={} row_dim={} rows={} hbb_rows={}",
            beta_dim,
            row_dim,
            assembled.rows.len(),
            assembled.hbb.nrows()
        ),
    );

    let start = Instant::now();
    let (_delta_t, _delta_beta, solve_diag) = assembled
        .solve_with_options(
            0.0,
            0.0,
            &ArrowSolveOptions::direct().with_ill_conditioning_tolerated(),
        )
        .map_err(|err| format!("inner solve failed: {err}"))?;
    print_stage(
        &fixture.shape,
        "inner_newton_solve",
        ms(start),
        &format!("pcg_iterations={}", solve_diag.iterations),
    );

    if fixture.shape.name == "qwen" {
        run_device_inner_iter(&fixture.shape)?;
    }

    let mut term_for_criterion = fixture.term.clone();
    let start = Instant::now();
    let (criterion, loss) = term_for_criterion.reml_criterion(
        fixture.target.view(),
        &fixture.rho,
        None,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    )?;
    print_stage(
        &fixture.shape,
        "criterion_eval",
        ms(start),
        &format!(
            "criterion={criterion:.12e} loss_total={:.12e}",
            loss.total()
        ),
    );

    let init_rho_flat = fixture.rho.to_flat();
    let n_params = init_rho_flat.len();
    let outer = SaeManifoldOuterObjective::new(
        fixture.term.clone(),
        fixture.target.clone(),
        None,
        fixture.rho.clone(),
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    let mut counted = CountingObjective::new(outer);
    let problem = OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .with_max_iter(OUTER_MAX_ITER);
    let start = Instant::now();
    let result = problem
        .run(&mut counted, "SAE perf harness")
        .map_err(|err| format!("outer fit failed: {err}"))?;
    print_stage(
        &fixture.shape,
        "outer_fit",
        ms(start),
        &format!(
            "final_value={:.12e} iterations={} converged={} eval_cost={} eval={} eval_efs={} seed_inner_state={}",
            result.final_value,
            result.iterations,
            result.converged,
            counted.eval_cost_count,
            counted.eval_count,
            counted.eval_efs_count,
            counted.seed_count
        ),
    );

    print_stage(&fixture.shape, "total", ms(total_start), "");
    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "sae_perf_harness".to_string());
    let Some(shape_name) = args.next() else {
        eprintln!("usage: {program} <tiny|color|qwen>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <tiny|color|qwen>");
        return ExitCode::from(2);
    }
    let Some(shape) = shape_named(&shape_name) else {
        eprintln!("unknown shape '{shape_name}'; expected tiny, color, or qwen");
        return ExitCode::from(2);
    };
    match run(shape) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sae_perf_harness: {err}");
            ExitCode::from(1)
        }
    }
}
