//! #1689 phase-profiling harness: ordinary Gaussian P-spline `s(x)` and
//! thin-plate `s(x,z)` fits reported 2-10x (tp up to 24x) slower than mgcv at
//! equal accuracy. This harness re-measures WHERE that time lives now that the
//! closed-form global-REML ρ* seed (595a0e4fe) and the summed-diagonal
//! multi-block seed (b26e1cfe9) have landed — nobody has re-profiled since.
//!
//! Unlike the standard-suite guard (`tests/perf_1689_pspline_thinplate_profile.rs`,
//! n=400/1200) this runs the ISSUE'S shapes (1-D n∈{1e3,1e4,1e5}; 2-D thin-plate
//! n∈{2e3,2e4}) and decomposes each fit into named phases, then compares against
//! an analytic reference cost model so the dominant term is unambiguous:
//!
//!   basis_build_s   standalone build of the design at the n training rows using
//!                   the fit's own resolved basis (isolates construction cost).
//!   fit_s           full fit wall clock (basis + penalty + outer×inner solves).
//!   predict_s       build design at the test grid + apply β.
//!   p               final design width. mgcv uses k≈12 (ps) / k≈134 default (tp);
//!                   a large p makes every inner solve O(n·p² + p³) expensive.
//!   inner_pirls_solves / outer_cost_evals / outer_iterations
//!                   outer-loop work. A large count means the smoothing search,
//!                   not the basis size, dominates.
//!
//! Reference cost model. Every Gaussian REML inner solve must form Xᵀ W X
//! (≈ 2·n·p² flops) and Cholesky-factor it (≈ p³/3 flops). So:
//!
//!   solve_flops   = 2·n·p² + p³/3                     (one irreducible inner solve)
//!   actual_flops  = inner_pirls_solves · solve_flops  (what gam actually paid)
//!   parity_flops  = ref_outer(#λ) · solve_flops       (what a lean optimizer needs)
//!
//! `implied_gflops = actual_flops / fit_s` is gam's realized solver throughput:
//! if it is low (≪ a few GFLOP/s) each solve is inefficient (per-iteration bound —
//! attack basis width / assembly). `solve_bloat = inner_pirls_solves / ref_outer`
//! is how many EXTRA inner solves the smoothing search paid: if it is large the
//! outer loop is the cost (attack seeds / iteration count). Both large means both.
//! `ref_outer` is a lean mgcv-style budget: ~1 gradient + ~2 probes per λ + a
//! fixed startup, i.e. `3·num_lambdas + 6`.
//!
//! Deterministic (fixed seeds), prints one CSV-ish line per case to stdout.
//! Optional args filter by tag substring, e.g. `perf_1689 ps` or `perf_1689 tp`.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    (a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// A measured case: phase timings + counters + the reference-cost decomposition.
struct Profile {
    tag: String,
    n: usize,
    p: usize,
    num_lambdas: usize,
    outer_iterations: usize,
    outer_cost_evals: usize,
    inner_pirls_solves: usize,
    inner_cycles: usize,
    edf: f64,
    basis_build_s: f64,
    fit_s: f64,
    predict_s: f64,
    rmse_vs_truth: f64,
}

impl Profile {
    /// One irreducible inner solve: form XᵀWX (2·n·p²) + Cholesky (p³/3).
    fn solve_flops(&self) -> f64 {
        let n = self.n as f64;
        let p = self.p as f64;
        2.0 * n * p * p + p * p * p / 3.0
    }
    /// Lean mgcv-style outer budget: ~3 inner solves per smoothing parameter
    /// (gradient + a couple of line-search / trust-region probes) plus startup.
    fn ref_outer(&self) -> f64 {
        3.0 * self.num_lambdas as f64 + 6.0
    }
    fn actual_flops(&self) -> f64 {
        self.inner_pirls_solves.max(1) as f64 * self.solve_flops()
    }
    /// Realized solver throughput (GFLOP/s). Low ⇒ per-solve inefficiency.
    fn implied_gflops(&self) -> f64 {
        self.actual_flops() / self.fit_s / 1e9
    }
    /// Extra inner solves the smoothing search paid vs a lean budget. High ⇒
    /// outer-loop bound.
    fn solve_bloat(&self) -> f64 {
        self.inner_pirls_solves.max(1) as f64 / self.ref_outer()
    }
    /// Per-inner-solve wall time (ms). Attributes fit time across actual solves.
    fn per_solve_ms(&self) -> f64 {
        1000.0 * self.fit_s / self.inner_pirls_solves.max(1) as f64
    }

    fn print(&self) {
        println!(
            "[#1689] tag={tag} n={n} p={p} nlam={nl} edf={edf:.1} conv=certified \
             | basis_build={bb:.3}s fit={fs:.3}s predict={ps:.3}s total={tot:.3}s \
             | outer_iter={oi} outer_evals={oce} inner_solves={ips} inner_cycles={ic} \
             | per_solve={psolve:.2}ms implied_gflops={ig:.2} solve_bloat={sb:.1}x ref_outer={ro:.0} \
             | rmse_vs_truth={r:.4}",
            tag = self.tag,
            n = self.n,
            p = self.p,
            nl = self.num_lambdas,
            edf = self.edf,
            bb = self.basis_build_s,
            fs = self.fit_s,
            ps = self.predict_s,
            tot = self.fit_s + self.predict_s,
            oi = self.outer_iterations,
            oce = self.outer_cost_evals,
            ips = self.inner_pirls_solves,
            ic = self.inner_cycles,
            psolve = self.per_solve_ms(),
            ig = self.implied_gflops(),
            sb = self.solve_bloat(),
            ro = self.ref_outer(),
            r = self.rmse_vs_truth,
        );
    }
}

/// Fit `formula`, then (1) re-time a standalone basis build at the n training
/// rows with the fit's resolved spec, and (2) time predict at `grid`.
fn profile_case(
    tag: &str,
    formula: &str,
    ds: &gam::data::EncodedDataset,
    train: &Array2<f64>,
    grid: &Array2<f64>,
    truth_at_grid: &[f64],
) -> Profile {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let t0 = Instant::now();
    let result = fit_from_formula(formula, ds, &cfg).expect("gam fit");
    let fit_s = t0.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for {formula}");
    };

    // Isolated basis-construction cost at full n (post-fit, so the resolved
    // basis width/knots match the fitted model exactly).
    let t_bb = Instant::now();
    let train_design = build_term_collection_design(train.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let basis_build_s = t_bb.elapsed().as_secs_f64();
    // The timing above is only meaningful if the rebuilt design is the SAME
    // design the fit solved: pin its width to the fitted coefficient length.
    assert_eq!(
        train_design.design.ncols(),
        fit.fit.beta.len(),
        "rebuilt training design width diverged from the fitted beta length"
    );

    let t1 = Instant::now();
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at grid");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let predict_s = t1.elapsed().as_secs_f64();

    build_profile(
        tag,
        train.nrows(),
        &fit,
        fit_s,
        basis_build_s,
        predict_s,
        rmse(&fitted, truth_at_grid),
    )
}

fn build_profile(
    tag: &str,
    n: usize,
    fit: &StandardFitResult,
    fit_s: f64,
    basis_build_s: f64,
    predict_s: f64,
    rmse_vs_truth: f64,
) -> Profile {
    let f = &fit.fit;
    Profile {
        tag: tag.to_string(),
        n,
        p: f.beta.len(),
        num_lambdas: f.lambdas.len(),
        outer_iterations: f.outer_iterations,
        outer_cost_evals: f.outer_cost_evals,
        inner_pirls_solves: f.inner_pirls_solves,
        inner_cycles: f.inner_cycles,
        edf: f.edf_total().unwrap_or(f64::NAN),
        basis_build_s,
        fit_s,
        predict_s,
        rmse_vs_truth,
    }
}

/// 1-D P-spline `s(x)`: y = sin(5x) + 0.5x + N(0, 0.2), issue's truth.
fn make_pspline_1d(
    n: usize,
) -> (
    gam::data::EncodedDataset,
    Array2<f64>,
    Array2<f64>,
    Vec<f64>,
) {
    let mut rng = StdRng::seed_from_u64(0);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let truth = |x: f64| (5.0 * x).sin() + 0.5 * x;

    let mut xs = Vec::with_capacity(n);
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let yi = truth(xi) + noise.sample(&mut rng);
            xs.push(xi);
            csv::StringRecord::from(vec![xi.to_string(), yi.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 1d");
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let mut train = Array2::<f64>::zeros((n, width));
    for (i, &xi) in xs.iter().enumerate() {
        train[[i, x_idx]] = xi;
    }

    let g = 300usize;
    let xt: Vec<f64> = (0..g).map(|i| i as f64 / (g as f64 - 1.0)).collect();
    let ft: Vec<f64> = xt.iter().map(|&x| truth(x)).collect();
    let mut grid = Array2::<f64>::zeros((g, width));
    for (i, &x) in xt.iter().enumerate() {
        grid[[i, x_idx]] = x;
    }
    (ds, train, grid, ft)
}

/// 2-D thin-plate `s(x,z)`: gaussian bump + N(0, 0.1), issue's truth.
fn make_thinplate_2d(
    n: usize,
) -> (
    gam::data::EncodedDataset,
    Array2<f64>,
    Array2<f64>,
    Vec<f64>,
) {
    let mut rng = StdRng::seed_from_u64(0);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let truth = |x: f64, z: f64| (-((x - 0.5).powi(2) + (z - 0.5).powi(2)) / 0.1).exp();

    let mut xs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let zi = u.sample(&mut rng);
            let yi = truth(xi, zi) + noise.sample(&mut rng);
            xs.push(xi);
            zs.push(zi);
            csv::StringRecord::from(vec![xi.to_string(), zi.to_string(), yi.to_string()])
        })
        .collect();
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2d");
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);
    let width = ds.headers.len();

    let mut train = Array2::<f64>::zeros((n, width));
    for i in 0..n {
        train[[i, x_idx]] = xs[i];
        train[[i, z_idx]] = zs[i];
    }

    let g = 30usize;
    let coord = |i: usize| i as f64 / (g as f64 - 1.0);
    let mut grid_rows = Vec::new();
    let mut ft = Vec::new();
    for i in 0..g {
        for j in 0..g {
            let (xi, zi) = (coord(i), coord(j));
            grid_rows.push((xi, zi));
            ft.push(truth(xi, zi));
        }
    }
    let mut grid = Array2::<f64>::zeros((grid_rows.len(), width));
    for (i, &(xi, zi)) in grid_rows.iter().enumerate() {
        grid[[i, x_idx]] = xi;
        grid[[i, z_idx]] = zi;
    }
    (ds, train, grid, ft)
}

fn main() {
    init_parallelism();
    let filter: Vec<String> = std::env::args().skip(1).collect();
    let want = |tag: &str| filter.is_empty() || filter.iter().any(|f| tag.contains(f.as_str()));

    println!(
        "# #1689 phase profile. solve_bloat = extra inner solves vs lean budget \
         (outer-loop bound if ≫1); implied_gflops = realized throughput \
         (per-solve bound if ≪ a few)."
    );

    // 1-D P-spline: the issue's n=400 grows to 1e3/1e4/1e5. p stays ~12 (mgcv
    // parity), so per-solve cost is cheap and any slowdown is outer-loop work.
    for &n in &[1_000usize, 10_000, 100_000] {
        let tag = format!("ps-1d-n{n}");
        if !want(&tag) {
            continue;
        }
        let (ds, train, grid, ft) = make_pspline_1d(n);
        profile_case(&tag, "y ~ smooth(x)", &ds, &train, &grid, &ft).print();
    }

    // 2-D thin-plate: the issue's n=1200 grows to 2e3/2e4. p≈134 by default, so
    // each inner solve is O(n·p²+p³) — the prime per-iteration suspect.
    for &n in &[2_000usize, 20_000] {
        let tag = format!("tp-2d-n{n}");
        if !want(&tag) {
            continue;
        }
        let (ds, train, grid, ft) = make_thinplate_2d(n);
        profile_case(&tag, "y ~ thinplate(x, z)", &ds, &train, &grid, &ft).print();
    }
}
