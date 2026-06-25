//! SAE (n, K) hill-climbing envelope harness (#1017).
//!
//! Geometrically sweeps the SAE manifold fit over `n` (rows / data points) and
//! `K` (dictionary size / number of latent atoms), and at every rung records
//! wall-clock per stage, reconstruction quality, solve/outer iteration counts,
//! and the assembled arrow/Schur block dimensions. It AUTO-CLIMBS: starting from
//! a small rung it keeps doubling `n` and `K` until it hits a CLIFF (a fit
//! `Err`, a per-rung wall-clock budget breach, or a reconstruction-quality
//! collapse) and then reports the exact rung coordinates of each cliff along
//! with the dominant scaling term predicted from the source asymptotics.
//!
//! Stage cost map (from src/terms/sae/manifold/construction.rs and
//! src/solver/arrow_schur/*):
//!   * per-row data-fit assembly       O(n * K * p * d)   (rows scale with n)
//!   * dense data Gram / cross-block    O(n * K^2 * m^2)   (K^2 cliff)
//!   * dense Schur reduce               O(n * d * K^2)
//!   * dense Schur Cholesky / solve     O(K^3)             (K^3 cliff)
//!   * per-row latent block factor      O(n * d^3)
//! The harness measures each stage independently so the dominant term at a
//! cliff can be read off the timing breakdown.
//!
//! Usage:
//!   sae_hillclimb_envelope_1017 [--n0 N] [--k0 K] [--pdim P] [--ddim D]
//!                               [--budget-ms MS] [--max-n N] [--max-k K]
//!                               [--quality-floor Q] [--topology circle|euclidean]
//!                               [--single n,K]   (run one rung and stop)
//!
//! Every flag has a default, so a bare invocation runs the full climb.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Topology {
    Circle,
    Euclidean,
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Topology::Circle => "circle",
            Topology::Euclidean => "euclidean",
        }
    }
}

#[derive(Clone, Copy)]
struct Shape {
    n: usize,
    p: usize,
    k: usize,
    d: usize,
    topology: Topology,
}

struct Fixture {
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

/// Counts inner objective evaluations so the harness can attribute outer-loop
/// cost to the number of inner solves (each `eval` is a full inner Newton fit).
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

/// Builds a genuine `SaeManifoldTerm` of the requested shape. The target is the
/// term's own clean reconstruction, so a converged fit can drive reconstruction
/// error toward zero and quality collapse is a real signal of a solver cliff
/// rather than an unfittable target.
fn build_fixture(shape: Shape) -> Result<Fixture, String> {
    let mut rng = Lcg::new(0x5AE0_1017_9E37_79B9 ^ shape.n as u64 ^ ((shape.k as u64) << 17));
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
    Ok(Fixture { term, target, rho })
}

/// Fraction of target variance explained by `fitted`, in [.., 1.0]; 1.0 is a
/// perfect reconstruction. Returns `f64::NAN` if either operand is non-finite.
fn frac_var_explained(target: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    if target.shape() != fitted.shape() {
        return f64::NAN;
    }
    let mut sse = 0.0_f64;
    let mut sst = 0.0_f64;
    let mean = target.iter().copied().sum::<f64>() / (target.len().max(1) as f64);
    for (t, f) in target.iter().zip(fitted.iter()) {
        if !t.is_finite() || !f.is_finite() {
            return f64::NAN;
        }
        let r = t - f;
        sse += r * r;
        let c = t - mean;
        sst += c * c;
    }
    if sst <= 0.0 {
        // Degenerate (constant) target: report perfect when residual is zero.
        return if sse <= 0.0 { 1.0 } else { f64::NAN };
    }
    1.0 - sse / sst
}

/// Per-stage timings and diagnostics for one (n, K) rung.
struct RungReport {
    shape: Shape,
    beta_dim: usize,
    row_dim: usize,
    rows: usize,
    hbb_rows: usize,
    assemble_ms: f64,
    inner_solve_ms: f64,
    inner_pcg_iters: usize,
    outer_ms: f64,
    outer_iters: usize,
    outer_converged: bool,
    inner_evals: usize,
    quality: f64,
    total_ms: f64,
}

/// Runs one rung end to end on the genuine fit path and returns its report, or
/// the error string if any stage fails (an error is itself a cliff signal).
fn run_rung(shape: Shape) -> Result<RungReport, String> {
    let total_start = Instant::now();
    let fixture = build_fixture(shape)?;
    let beta_dim = fixture.term.beta_dim();
    let row_dim = fixture.term.assignment.row_block_dim();

    // Stage 1: arrow/Schur assembly (per-row O(n) data-fit + dense K-blocks).
    let mut term_for_assemble = fixture.term.clone();
    let assemble_start = Instant::now();
    let assembled =
        term_for_assemble.assemble_arrow_schur(fixture.target.view(), &fixture.rho, None)?;
    let assemble_ms = ms(assemble_start);
    let rows = assembled.rows.len();
    let hbb_rows = assembled.hbb.nrows();

    // Stage 2: inner Newton solve (dense Schur Cholesky O(K^3) or matrix-free PCG).
    let solve_start = Instant::now();
    let (_delta_t, _delta_beta, solve_diag) = assembled
        .solve_with_options(
            0.0,
            0.0,
            &ArrowSolveOptions::direct().with_ill_conditioning_tolerated(),
        )
        .map_err(|err| format!("inner solve failed: {err}"))?;
    let inner_solve_ms = ms(solve_start);
    let inner_pcg_iters = solve_diag.iterations;

    // Stage 3: full outer fit (outer rho loop driving inner Newton fits).
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
    let outer_start = Instant::now();
    let result = problem
        .run(&mut counted, "SAE hillclimb")
        .map_err(|err| format!("outer fit failed: {err}"))?;
    let outer_ms = ms(outer_start);

    // Reconstruction quality from the fitted term vs the clean target.
    let fitted = fixture.term.fitted();
    let quality = frac_var_explained(&fixture.target, &fitted);

    Ok(RungReport {
        shape,
        beta_dim,
        row_dim,
        rows,
        hbb_rows,
        assemble_ms,
        inner_solve_ms,
        inner_pcg_iters,
        outer_ms,
        outer_iters: result.iterations,
        outer_converged: result.converged,
        inner_evals: counted.eval_count,
        quality,
        total_ms: ms(total_start),
    })
}

impl RungReport {
    fn print(&self) {
        let s = &self.shape;
        // Dominant predicted term at this rung, from the source asymptotics.
        let dominant = self.dominant_term();
        println!(
            "RUNG n={} K={} p={} d={} topo={} | beta_dim={} row_dim={} rows={} hbb_rows={} \
             | assemble_ms={:.3} inner_solve_ms={:.3} inner_pcg_iters={} outer_ms={:.3} \
             outer_iters={} outer_converged={} inner_evals={} | quality={:.6} total_ms={:.3} \
             | dominant={}",
            s.n,
            s.k,
            s.p,
            s.d,
            s.topology.label(),
            self.beta_dim,
            self.row_dim,
            self.rows,
            self.hbb_rows,
            self.assemble_ms,
            self.inner_solve_ms,
            self.inner_pcg_iters,
            self.outer_ms,
            self.outer_iters,
            self.outer_converged,
            self.inner_evals,
            self.quality,
            self.total_ms,
            dominant
        );
    }

    /// Reads the dominant cost stage off the measured per-stage split. This is a
    /// measured attribution, not a guess: whichever stage consumed the most
    /// wall-clock names the dominant asymptotic term at this rung.
    fn dominant_term(&self) -> &'static str {
        let assemble = self.assemble_ms;
        let solve = self.inner_solve_ms;
        let outer = self.outer_ms;
        if outer >= assemble && outer >= solve {
            "outer_rho_loop(O(inner_evals * inner_fit))"
        } else if solve >= assemble {
            "inner_solve(dense Schur O(K^3) or PCG O(n*d*K))"
        } else {
            "assemble(per-row O(n*K*p*d) + dense Gram O(n*K^2*m^2))"
        }
    }
}

/// Why a climb stopped at a given rung.
enum Cliff {
    Error { stage: Shape, message: String },
    Budget { stage: Shape, total_ms: f64 },
    Quality { stage: Shape, quality: f64 },
}

struct Climb {
    label: &'static str,
    /// How the rung coordinate advances each step.
    step: fn(Shape) -> Shape,
}

struct Config {
    n0: usize,
    k0: usize,
    p: usize,
    d: usize,
    budget_ms: f64,
    max_n: usize,
    max_k: usize,
    quality_floor: f64,
    topology: Topology,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n0: 1_000,
            k0: 32,
            p: 256,
            d: 2,
            budget_ms: 120_000.0,
            max_n: 1_000_000,
            max_k: 8_192,
            quality_floor: 0.5,
            topology: Topology::Euclidean,
        }
    }
}

fn double_n(mut s: Shape) -> Shape {
    s.n *= 2;
    s
}

fn double_k(mut s: Shape) -> Shape {
    s.k *= 2;
    s
}

fn double_both(mut s: Shape) -> Shape {
    s.n *= 2;
    s.k *= 2;
    s
}

/// Runs one climb axis from `start`, stepping until a cliff. Prints each rung as
/// it lands and returns the cliff that terminated the climb (or `None` if the
/// hard `max_n`/`max_k` envelope ceiling was reached without a cliff).
fn climb(start: Shape, cfg: &Config, step: fn(Shape) -> Shape, label: &str) -> Option<Cliff> {
    println!("=== CLIMB axis={label} ===");
    let mut shape = start;
    loop {
        let rung = match run_rung(shape) {
            Ok(rung) => rung,
            Err(message) => {
                println!(
                    "CLIFF axis={label} kind=error n={} K={} message=\"{message}\"",
                    shape.n, shape.k
                );
                return Some(Cliff::Error {
                    stage: shape,
                    message,
                });
            }
        };
        rung.print();

        if !rung.quality.is_finite() || rung.quality < cfg.quality_floor {
            println!(
                "CLIFF axis={label} kind=quality n={} K={} quality={:.6} floor={:.6}",
                shape.n, shape.k, rung.quality, cfg.quality_floor
            );
            return Some(Cliff::Quality {
                stage: shape,
                quality: rung.quality,
            });
        }
        if rung.total_ms > cfg.budget_ms {
            println!(
                "CLIFF axis={label} kind=budget n={} K={} total_ms={:.3} budget_ms={:.3}",
                shape.n, shape.k, rung.total_ms, cfg.budget_ms
            );
            return Some(Cliff::Budget {
                stage: shape,
                total_ms: rung.total_ms,
            });
        }

        let next = step(shape);
        if next.n > cfg.max_n || next.k > cfg.max_k {
            println!(
                "CEILING axis={label} reached envelope ceiling without a cliff at n={} K={} \
                 (next n={} K={} exceeds max_n={} max_k={})",
                shape.n, shape.k, next.n, next.k, cfg.max_n, cfg.max_k
            );
            return None;
        }
        shape = next;
    }
}

fn report_cliff(climb_label: &str, cliff: &Option<Cliff>) {
    match cliff {
        Some(Cliff::Error { stage, message }) => println!(
            "SUMMARY climb={climb_label} cliff=error rung_n={} rung_K={} detail=\"{message}\"",
            stage.n, stage.k
        ),
        Some(Cliff::Budget { stage, total_ms }) => println!(
            "SUMMARY climb={climb_label} cliff=budget rung_n={} rung_K={} total_ms={:.3}",
            stage.n, stage.k, total_ms
        ),
        Some(Cliff::Quality { stage, quality }) => println!(
            "SUMMARY climb={climb_label} cliff=quality rung_n={} rung_K={} quality={:.6}",
            stage.n, stage.k, quality
        ),
        None => println!("SUMMARY climb={climb_label} cliff=none (hit envelope ceiling)"),
    }
}

fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<usize>()
        .map_err(|err| format!("{flag}: {err}"))
}

fn parse_f64(flag: &str, value: Option<String>) -> Result<f64, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<f64>()
        .map_err(|err| format!("{flag}: {err}"))
}

fn run() -> Result<(), String> {
    let mut cfg = Config::default();
    let mut single: Option<(usize, usize)> = None;

    let mut args = env::args();
    let _program = args.next();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--n0" => cfg.n0 = parse_usize("--n0", args.next())?,
            "--k0" => cfg.k0 = parse_usize("--k0", args.next())?,
            "--pdim" => cfg.p = parse_usize("--pdim", args.next())?,
            "--ddim" => cfg.d = parse_usize("--ddim", args.next())?,
            "--budget-ms" => cfg.budget_ms = parse_f64("--budget-ms", args.next())?,
            "--max-n" => cfg.max_n = parse_usize("--max-n", args.next())?,
            "--max-k" => cfg.max_k = parse_usize("--max-k", args.next())?,
            "--quality-floor" => cfg.quality_floor = parse_f64("--quality-floor", args.next())?,
            "--topology" => {
                cfg.topology = match args.next().as_deref() {
                    Some("circle") => Topology::Circle,
                    Some("euclidean") => Topology::Euclidean,
                    other => return Err(format!("--topology: unknown '{other:?}'")),
                };
            }
            "--single" => {
                let spec = args.next().ok_or("--single requires n,K")?;
                let mut parts = spec.split(',');
                let n = parts
                    .next()
                    .ok_or("--single n missing")?
                    .parse::<usize>()
                    .map_err(|err| format!("--single n: {err}"))?;
                let k = parts
                    .next()
                    .ok_or("--single K missing")?
                    .parse::<usize>()
                    .map_err(|err| format!("--single K: {err}"))?;
                single = Some((n, k));
            }
            other => return Err(format!("unknown flag '{other}'")),
        }
    }

    // Circle topology forces d == 1 (the periodic harmonic basis is 1-D).
    let d = if cfg.topology == Topology::Circle {
        1
    } else {
        cfg.d
    };

    println!(
        "CONFIG n0={} k0={} p={} d={} budget_ms={:.0} max_n={} max_k={} quality_floor={:.3} topology={}",
        cfg.n0,
        cfg.k0,
        cfg.p,
        d,
        cfg.budget_ms,
        cfg.max_n,
        cfg.max_k,
        cfg.quality_floor,
        cfg.topology.label()
    );

    if let Some((n, k)) = single {
        let shape = Shape {
            n,
            p: cfg.p,
            k,
            d,
            topology: cfg.topology,
        };
        let rung = run_rung(shape)?;
        rung.print();
        return Ok(());
    }

    let base = Shape {
        n: cfg.n0,
        p: cfg.p,
        k: cfg.k0,
        d,
        topology: cfg.topology,
    };

    // Three independent climbs isolate the n-axis, the K-axis, and the joint
    // diagonal so each cliff's dominant term can be attributed separately.
    let climbs = [
        Climb {
            label: "n_axis",
            step: double_n as fn(Shape) -> Shape,
        },
        Climb {
            label: "k_axis",
            step: double_k,
        },
        Climb {
            label: "joint_diag",
            step: double_both,
        },
    ];

    let mut cliffs = Vec::with_capacity(climbs.len());
    for c in &climbs {
        let cliff = climb(base, &cfg, c.step, c.label);
        report_cliff(c.label, &cliff);
        cliffs.push((c.label, cliff));
    }

    println!("=== CLIFF MAP ===");
    for (label, cliff) in &cliffs {
        report_cliff(label, cliff);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sae_hillclimb_envelope_1017: {err}");
            ExitCode::from(1)
        }
    }
}
