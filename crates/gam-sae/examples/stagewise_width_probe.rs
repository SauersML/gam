//! Stagewise (SAC) WIDTH-SCALING PROBE — localizes the p-wall that hung the
//! Qwen 3.6 (q36b) L17 T2 smoke (job 12682738: 4k rows, p=2048, circle,
//! max_births=3 — no output for the whole job wall).
//!
//! Synthetic fixture at arbitrary (n, p): k planted circles in disjoint
//! axis-aligned 2-planes plus an optional power-law "clutter tail" of extra
//! above-noise directions (the real T1 residual is NOT k clean planes — the
//! ISA producer's above-floor subspace scales with the tail, and any cost
//! super-linear in that subspace only shows on a tailed fixture).
//!
//! Every stagewise event is timestamped through the #2138 progress callback,
//! and the callback aborts the fit cleanly once `--budget-s` is exceeded, so
//! a wall never eats the job: we always get the last completed phase.
//!
//! Usage:
//!   stagewise_width_probe <out.json> [--rows 2000] [--p 512] [--circles 4]
//!     [--tail 64] [--noise 0.05] [--max-births 3] [--inner 24]
//!     [--budget-s 900] [--seed 1]

use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    LatentManifold, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    StagewiseConfig, StagewiseProgress, fit_stagewise,
};
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

fn lcg_normal(s: &mut u64) -> f64 {
    let (u1, u2) = (lcg(s).max(1e-12), lcg(s));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

struct Fixture {
    data: Array2<f64>,
    turns0: Array2<f64>,
}

/// `k` unit-amplitude-decaying circles on axes `(2c, 2c+1)`, then `tail` extra
/// random-direction components with amplitudes `0.5 · (j+1)^{-0.7}` (above the
/// noise floor, below the circles), then isotropic noise `sigma`.
fn fixture(n: usize, p: usize, k: usize, tail: usize, sigma: f64, seed: u64) -> Fixture {
    assert!(p >= 2 * k, "p must be >= 2*circles");
    let mut s = seed;
    let mut data = Array2::<f64>::zeros((n, p));
    let mut turns0 = Array2::<f64>::zeros((n, 1));
    let amps: Vec<f64> = (0..k)
        .map(|c| 1.0 - 0.45 * (c as f64) / ((k.max(2) - 1) as f64))
        .collect();
    // Clutter tail directions: fixed random unit vectors, one scalar loading per row.
    let mut tail_dirs = Array2::<f64>::zeros((tail, p));
    for j in 0..tail {
        let mut norm = 0.0;
        for d in 0..p {
            let g = lcg_normal(&mut s);
            tail_dirs[[j, d]] = g;
            norm += g * g;
        }
        let norm = norm.sqrt().max(1e-12);
        for d in 0..p {
            tail_dirs[[j, d]] /= norm;
        }
    }
    for i in 0..n {
        for (c, amp) in amps.iter().enumerate() {
            let t = lcg(&mut s);
            if c == 0 {
                turns0[[i, 0]] = t;
            }
            let ang = std::f64::consts::TAU * t;
            data[[i, 2 * c]] += amp * ang.cos();
            data[[i, 2 * c + 1]] += amp * ang.sin();
        }
        for j in 0..tail {
            let a = 0.5 * ((j + 1) as f64).powf(-0.7) * lcg_normal(&mut s);
            for d in 0..p {
                data[[i, d]] += a * tail_dirs[[j, d]];
            }
        }
        for d in 0..p {
            data[[i, d]] += sigma * lcg_normal(&mut s);
        }
    }
    Fixture { data, turns0 }
}

/// K=1 circle seed on circle 0's (noisily known) coordinate — mirrors the
/// #2111 fixture seeding so the probe times SAC, not cold-start seeding.
fn seed_term(turns0: &Array2<f64>, p: usize) -> (SaeManifoldTerm, SaeManifoldRho) {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi, jet) = evaluator.evaluate(turns0.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let atom = SaeManifoldAtom::new(
        "probe_seed_c0".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let n = turns0.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![turns0.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .ok_or("usage: stagewise_width_probe <out.json> [--rows N] [--p P] ...")?;
    let n: usize = arg(&args, "--rows", 2000);
    let p: usize = arg(&args, "--p", 512);
    let k: usize = arg(&args, "--circles", 4);
    let tail: usize = arg(&args, "--tail", 64);
    let sigma: f64 = arg(&args, "--noise", 0.05);
    let max_births: usize = arg(&args, "--max-births", 3);
    let inner: usize = arg(&args, "--inner", 24);
    let budget_s: f64 = arg(&args, "--budget-s", 900.0);
    let seed: u64 = arg(&args, "--seed", 1);

    println!(
        "[probe] n={n} p={p} circles={k} tail={tail} noise={sigma} max_births={max_births} \
         inner={inner} budget_s={budget_s}"
    );
    let t0 = Instant::now();
    let fx = fixture(n, p, k, tail, sigma, seed);
    println!(
        "[probe] fixture built in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let config = StagewiseConfig {
        inner_max_iter: inner,
        learning_rate: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        max_births,
        max_backfit_sweeps: 3,
        min_effect_ev: 0.0,
        max_factor_rank: 4,
        structured_whitening: false,
    };

    let (mut term, mut rho) = seed_term(&fx.turns0, p);
    let t_seed = Instant::now();
    term.run_joint_fit_arrow_schur(
        fx.data.view(),
        &mut rho,
        None,
        config.inner_max_iter,
        config.learning_rate,
        config.ridge_ext_coord,
        config.ridge_beta,
    )?;
    let seed_fit_s = t_seed.elapsed().as_secs_f64();
    println!("[probe] K=1 seed joint fit: {seed_fit_s:.2}s");

    let mut events: Vec<Value> = Vec::new();
    let t_run = Instant::now();
    let mut last = Instant::now();
    let mut callback = |pr: StagewiseProgress<'_>| -> Result<(), String> {
        let now = Instant::now();
        let rec = json!({
            "event": format!("{:?}", pr.event),
            "birth_round": pr.birth_round,
            "backfit_sweep": pr.backfit_sweep,
            "candidate": pr.candidate.map(|c| format!("{c:?}")),
            "accepted": pr.accepted,
            "k_atoms": pr.k_atoms,
            "ev": pr.ev,
            "since_prev_s": (now - last).as_secs_f64(),
            "elapsed_s": (now - t_run).as_secs_f64(),
        });
        println!(
            "[probe +{:8.2}s] (+{:7.2}s) {:<24} round={} k={} ev={:?}",
            (now - t_run).as_secs_f64(),
            (now - last).as_secs_f64(),
            format!("{:?}", pr.event),
            pr.birth_round,
            pr.k_atoms,
            pr.ev,
        );
        std::io::stdout().flush().ok();
        events.push(rec);
        last = now;
        if (now - t_run).as_secs_f64() > budget_s {
            return Err(format!(
                "PROBE_BUDGET_EXCEEDED after {:.1}s at {:?} (round {})",
                (now - t_run).as_secs_f64(),
                pr.event,
                pr.birth_round
            ));
        }
        Ok(())
    };
    let mut cb: &mut dyn for<'e> FnMut(StagewiseProgress<'e>) -> Result<(), String> = &mut callback;
    let result = fit_stagewise(
        term,
        rho,
        fx.data.view(),
        None,
        None,
        &config,
        Some(&mut cb),
        None,
    );
    let total_s = t_run.elapsed().as_secs_f64();
    let (status, k_final, ev_trace, stop) = match &result {
        Ok(r) => (
            "completed",
            r.term.k_atoms() as i64,
            json!(r.report.ev_trace),
            format!("{:?}", r.report.stopped_reason),
        ),
        Err(e) => ("aborted", -1, json!(null), e.clone()),
    };
    println!("[probe] stagewise {status} in {total_s:.2}s: {stop}");

    let report = json!({
        "n": n, "p": p, "circles": k, "tail": tail, "noise": sigma,
        "max_births": max_births, "inner_max_iter": inner, "seed": seed,
        "seed_fit_s": seed_fit_s,
        "stagewise_s": total_s,
        "status": status,
        "stop": stop,
        "k_final": k_final,
        "ev_trace": ev_trace,
        "events": events,
    });
    std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {out_path}: {e}"))?;
    println!("[probe] wrote {out_path}");
    Ok(())
}
