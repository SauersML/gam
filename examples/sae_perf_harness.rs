use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gam::gpu::kernels::arrow_schur::solve_reduced_beta_pcg_with_diagnostics;
use gam::gpu::kernels::sae_resident::{
    DeviceResidentArrowError, DeviceResidentInnerOptions, qwen_non_gating_fixture,
    qwen_non_gating_fixture_seeded, run_resident_fits_multiplexed, run_resident_fits_sequential,
};
use gam::solver::arrow_schur::ArrowSolveOptions;
use gam::solver::estimate::EstimationError;
use gam::solver::rho_optimizer::{
    EfsEval, OuterCapability, OuterEval, OuterObjective, OuterProblem, SeedOutcome,
};
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};

const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const LOG_LAMBDA_SPARSE: f64 = -12.0;
const LOG_LAMBDA_SMOOTH: f64 = -12.0;
const INNER_MAX_ITER: usize = 12;
const OUTER_MAX_ITER: usize = 1;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
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
        false
    }

    fn requires_continuation_path_entry(&self) -> bool {
        false
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
    let target = term.fitted();
    let rho = SaeManifoldRho::new(
        LOG_LAMBDA_SPARSE,
        LOG_LAMBDA_SMOOTH,
        vec![Array1::<f64>::zeros(0); shape.k],
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

fn dense_spd_for_device_pcg(k: usize) -> (Array2<f64>, Array1<f64>) {
    let mut s = Array2::<f64>::zeros((k, k));
    for row in 0..k {
        s[[row, row]] = 2.5 + 0.001 * ((row % 17) as f64);
        if row + 1 < k {
            s[[row, row + 1]] = -0.05;
            s[[row + 1, row]] = -0.05;
        }
        if row + 7 < k {
            s[[row, row + 7]] = 0.01;
            s[[row + 7, row]] = 0.01;
        }
    }
    let rhs = Array1::from_shape_fn(k, |idx| ((idx as f64 + 1.0) * 0.013).sin());
    (s, rhs)
}

fn dense_matvec_ref(s: &Array2<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
    let k = x.len();
    out.fill(0.0);
    for row in 0..k {
        let mut acc = 0.0;
        for col in 0..k {
            acc += s[[row, col]] * x[col];
        }
        out[row] = acc;
    }
}

fn cpu_jacobi_pcg_ref(
    s: &Array2<f64>,
    rhs: &Array1<f64>,
    max_iterations: usize,
    relative_tolerance: f64,
) -> Array1<f64> {
    let k = rhs.len();
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if rhs_norm == 0.0 {
        return Array1::<f64>::zeros(k);
    }
    let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);
    let inv_diag: Vec<f64> = (0..k).map(|idx| 1.0 / s[[idx, idx]]).collect();
    let mut x = Array1::<f64>::zeros(k);
    let mut r = rhs.clone();
    let mut z = Array1::from_shape_fn(k, |idx| inv_diag[idx] * r[idx]);
    let mut p = z.clone();
    let mut sp = Array1::<f64>::zeros(k);
    let mut rz = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
    for _ in 0..max_iterations.max(1) {
        dense_matvec_ref(s, &p, &mut sp);
        let p_sp = p.iter().zip(sp.iter()).map(|(a, b)| a * b).sum::<f64>();
        let alpha = rz / p_sp;
        for idx in 0..k {
            x[idx] += alpha * p[idx];
            r[idx] -= alpha * sp[idx];
        }
        let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm <= tol {
            break;
        }
        for idx in 0..k {
            z[idx] = inv_diag[idx] * r[idx];
        }
        let rz_next = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
        let beta = rz_next / rz;
        for idx in 0..k {
            p[idx] = z[idx] + beta * p[idx];
        }
        rz = rz_next;
    }
    x
}

fn run_device_pcg(shape: &Shape) -> Result<(), String> {
    let k = shape.p.max(512);
    let (s, rhs) = dense_spd_for_device_pcg(k);
    let max_iterations = 200usize;
    let relative_tolerance = 1.0e-12;

    let cpu_start = Instant::now();
    let cpu = cpu_jacobi_pcg_ref(&s, &rhs, max_iterations, relative_tolerance);
    let cpu_ms = ms(cpu_start);

    let device_start = Instant::now();
    let (device, diag) =
        match solve_reduced_beta_pcg_with_diagnostics(&s, &rhs, max_iterations, relative_tolerance)
        {
            Ok(out) => out,
            Err(err) => {
                print_stage(
                    shape,
                    "device_pcg",
                    ms(device_start),
                    &format!(
                        "status=skipped reason=\"{:?}\" reduced_k={} cpu_ms={cpu_ms:.3}",
                        err, k
                    ),
                );
                return Ok(());
            }
        };
    let device_ms = ms(device_start);
    let max_err = max_abs_diff(
        cpu.as_slice().ok_or("CPU PCG output not contiguous")?,
        device
            .as_slice()
            .ok_or("device PCG output not contiguous")?,
    );
    print_stage(
        shape,
        "device_pcg",
        device_ms,
        &format!(
            "status=ok reduced_k={} cpu_ms={cpu_ms:.3} speedup={:.3} max_abs_err={max_err:.3e} iterations={} matvec_calls={} precond_apply_calls={} final_rel_residual={:.3e}",
            k,
            cpu_ms / device_ms.max(f64::MIN_POSITIVE),
            diag.iterations,
            diag.matvec_calls,
            diag.precond_apply_calls,
            diag.final_relative_residual
        ),
    );
    if max_err > DEVICE_PARITY_TOL {
        return Err(format!(
            "device_pcg parity failed: max_abs_err={max_err:e} > {DEVICE_PARITY_TOL:e}"
        ));
    }
    Ok(())
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
            "status=ok build_ms={build_ms:.3} cpu_ms={cpu_ms:.3} speedup={:.3} max_abs_step_err={max_err:.3e} objective={:.6e} grad_norm={:.6e} execution_path={} host_bytes={} device_bytes={}",
            cpu_ms / device_ms.max(f64::MIN_POSITIVE),
            device.objective,
            device.gradient_norm,
            device.execution_path.as_str(),
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

fn run_device_fit(shape: &Shape) -> Result<(), String> {
    let build_start = Instant::now();
    let workspace = qwen_non_gating_fixture().map_err(|err| err.to_string())?;
    let build_ms = ms(build_start);
    let opts = DeviceResidentInnerOptions::default();

    let cpu_start = Instant::now();
    let cpu = workspace
        .cpu_reference_fit(&opts)
        .map_err(|err| format!("device_fit CPU reference failed: {err}"))?;
    let cpu_ms = ms(cpu_start);

    let device_start = Instant::now();
    let device = match workspace.device_fit(&opts) {
        Ok(outcome) => outcome,
        Err(DeviceResidentArrowError::Unavailable { reason }) => {
            print_stage(
                shape,
                "device_fit",
                ms(device_start),
                &format!(
                    "status=skipped reason=\"{}\" build_ms={build_ms:.3} cpu_ms={cpu_ms:.3} cpu_iters={} cpu_converged={}",
                    reason.replace('"', "'"),
                    cpu.iterations,
                    cpu.converged
                ),
            );
            return Ok(());
        }
        Err(err) => return Err(format!("device_fit device solve failed: {err}")),
    };
    let device_ms = ms(device_start);

    let t_err = max_abs_diff(
        cpu.t.as_slice().ok_or("CPU t not contiguous")?,
        device.t.as_slice().ok_or("device t not contiguous")?,
    );
    let beta_err = max_abs_diff(
        cpu.beta.as_slice().ok_or("CPU beta not contiguous")?,
        device.beta.as_slice().ok_or("device beta not contiguous")?,
    );
    let obj_err = (cpu.objective - device.objective).abs();
    let max_err = t_err.max(beta_err).max(obj_err);
    print_stage(
        shape,
        "device_fit",
        device_ms,
        &format!(
            "status=ok build_ms={build_ms:.3} cpu_ms={cpu_ms:.3} speedup={:.3} max_abs_err={max_err:.3e} iters={} accepted={} converged={} objective={:.6e} grad_norm={:.6e} execution_path={}",
            cpu_ms / device_ms.max(f64::MIN_POSITIVE),
            device.iterations,
            device.accepted_iterations,
            device.converged,
            device.objective,
            device.gradient_norm,
            device.execution_path.as_str()
        ),
    );
    if max_err > DEVICE_PARITY_TOL {
        return Err(format!(
            "device_fit parity failed: max_abs_err={max_err:e} > {DEVICE_PARITY_TOL:e}"
        ));
    }

    // Phase 4: stream-multiplexed independent fits == sequential, bit-identical.
    let fit_count = 8usize;
    let mut workspaces = Vec::with_capacity(fit_count);
    let mut seq_workspaces = Vec::with_capacity(fit_count);
    for idx in 0..fit_count {
        let seed = 0x1017_0004_0000_0001u64.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9));
        workspaces.push(qwen_non_gating_fixture_seeded(seed).map_err(|err| err.to_string())?);
        seq_workspaces.push(qwen_non_gating_fixture_seeded(seed).map_err(|err| err.to_string())?);
    }

    let seq_start = Instant::now();
    let sequential = run_resident_fits_sequential(&seq_workspaces, &opts);
    let seq_ms = ms(seq_start);
    if let Some(Err(DeviceResidentArrowError::Unavailable { reason })) = sequential.first() {
        print_stage(
            shape,
            "device_multiplex",
            0.0,
            &format!(
                "status=skipped reason=\"{}\" fits={fit_count}",
                reason.replace('"', "'")
            ),
        );
        return Ok(());
    }

    let mux_start = Instant::now();
    let multiplexed = run_resident_fits_multiplexed(workspaces, opts)?;
    let mux_ms = ms(mux_start);

    let mut mux_max_err = 0.0_f64;
    for (seq, mux) in sequential.iter().zip(multiplexed.iter()) {
        let seq_fit = seq
            .as_ref()
            .map_err(|err| format!("sequential fit failed: {err}"))?;
        let mux_fit = mux
            .as_ref()
            .map_err(|err| format!("multiplexed fit failed: {err}"))?;
        let te = max_abs_diff(
            seq_fit.outcome.t.as_slice().ok_or("seq t not contiguous")?,
            mux_fit.outcome.t.as_slice().ok_or("mux t not contiguous")?,
        );
        let be = max_abs_diff(
            seq_fit
                .outcome
                .beta
                .as_slice()
                .ok_or("seq beta not contiguous")?,
            mux_fit
                .outcome
                .beta
                .as_slice()
                .ok_or("mux beta not contiguous")?,
        );
        let oe = (seq_fit.outcome.objective - mux_fit.outcome.objective).abs();
        mux_max_err = mux_max_err.max(te.max(be).max(oe));
    }
    print_stage(
        shape,
        "device_multiplex",
        mux_ms,
        &format!(
            "status=ok fits={fit_count} seq_ms={seq_ms:.3} mux_ms={mux_ms:.3} speedup={:.3} mux_vs_seq_max_abs_err={mux_max_err:.3e}",
            seq_ms / mux_ms.max(f64::MIN_POSITIVE)
        ),
    );
    if mux_max_err != 0.0 {
        return Err(format!(
            "device_multiplex non-identical to sequential: max_abs_err={mux_max_err:e} (independent fits must be bit-identical)"
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
        run_device_pcg(&fixture.shape)?;
        run_device_inner_iter(&fixture.shape)?;
        run_device_fit(&fixture.shape)?;
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
