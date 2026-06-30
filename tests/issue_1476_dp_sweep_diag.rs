//! DIAGNOSTIC (temporary): wide dp-toggle sweep for #1476 to characterize the
//! current failure rate of the double-penalty over-shrink under concurvity.
//! Not a permanent gate — prints per-seed ratios across 16 seeds and asserts a
//! deliberately loose bound so it always reports.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

struct EprintLogger;
impl log::Log for EprintLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        let msg = format!("{}", record.args());
        if msg.contains("[OUTER]")
            || msg.contains("[SEED-GRID]")
            || msg.contains("#1371")
            || msg.contains("keep-best")
            || msg.contains("candidate")
            || msg.contains("seed ")
        {
            eprintln!("LOG {} {}", record.level(), msg);
        }
    }
    fn flush(&self) {}
}
static LOGGER: EprintLogger = EprintLogger;
fn init_log() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);
}

const N: usize = 400;
const RHO: f64 = 0.90;
const NOISE: f64 = 0.30;

fn encode(cols: &[(&str, &[f64])]) -> EncodedDataset {
    let n = cols[0].1.len();
    let headers: Vec<String> = cols.iter().map(|(h, _)| (*h).to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(
                cols.iter()
                    .map(|(_, c)| c[i].to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn rank_to_unit(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).expect("finite"));
    let mut rank = vec![0usize; n];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    rank.iter().map(|&r| (r as f64 + 0.5) / n as f64).collect()
}

fn fit_and_score(x1: &[f64], x2: &[f64], y: &[f64], mu: &[f64], dp: bool) -> (f64, f64) {
    fit_and_score_diag(x1, x2, y, mu, dp, false)
}

fn fit_and_score_diag(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    mu: &[f64],
    dp: bool,
    dump: bool,
) -> (f64, f64) {
    let ds = encode(&[("x1", x1), ("x2", x2), ("y", y)]);
    let col = ds.column_map();
    let (i1, i2) = (col["x1"], col["x2"]);
    let width = ds.headers.len();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = if dp {
        "y ~ s(x1) + s(x2)".to_string()
    } else {
        "y ~ s(x1, double_penalty=False) + s(x2, double_penalty=False)".to_string()
    };
    let result = fit_from_formula(&formula, &ds, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("standard fit");
    };
    if dump {
        let ll: Vec<f64> = fit.fit.log_lambdas.iter().copied().collect();
        eprintln!(
            "  [dump dp={dp}] log_lambdas={:?}  outer_iters={} converged={}",
            ll.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
            fit.fit.outer_iterations,
            fit.fit.outer_converged
        );
    }
    let edf = fit.fit.edf_total().expect("edf");
    let n = x1.len();
    let mut pts = Array2::<f64>::zeros((n, width));
    for r in 0..n {
        pts[[r, i1]] = x1[r];
        pts[[r, i2]] = x2[r];
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec).expect("design");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    (rmse(&fitted, mu), edf)
}

fn make_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nrm = Normal::new(0.0, 1.0).expect("normal");
    let mut g0 = Vec::with_capacity(N);
    let mut g1 = Vec::with_capacity(N);
    for _ in 0..N {
        let z0: f64 = nrm.sample(&mut rng);
        let z1: f64 = nrm.sample(&mut rng);
        g0.push(z0);
        g1.push(RHO * z0 + (1.0 - RHO * RHO).sqrt() * z1);
    }
    let x1 = rank_to_unit(&g0);
    let x2 = rank_to_unit(&g1);
    let f2: Vec<f64> = x2.iter().map(|&v| v * v).collect();
    let f2_mean = f2.iter().sum::<f64>() / N as f64;
    let mu: Vec<f64> = (0..N)
        .map(|i| (TAU * x1[i]).sin() + (f2[i] - f2_mean))
        .collect();
    let y: Vec<f64> = mu.iter().map(|&m| m + NOISE * nrm.sample(&mut rng)).collect();
    (x1, x2, y, mu)
}

#[test]
fn dp_sweep_diag_16_seeds() {
    init_parallelism();
    let mut ratios = Vec::new();
    let mut n_bad = 0;
    for k in 0..16u64 {
        let seed = 1_476_011 + k;
        let (x1, x2, y, mu) = make_data(seed);
        let (err_dp, edf_dp) = fit_and_score(&x1, &x2, &y, &mu, true);
        let (err_sp, edf_sp) = fit_and_score(&x1, &x2, &y, &mu, false);
        let ratio = err_dp / err_sp.max(1e-12);
        if ratio > 1.25 {
            n_bad += 1;
        }
        eprintln!(
            "seed {seed}: dp rmse={err_dp:.5} (edf {edf_dp:.2}) | sp rmse={err_sp:.5} (edf {edf_sp:.2}) | ratio={ratio:.3}{}",
            if ratio > 1.25 { "  <-- BAD" } else { "" }
        );
        ratios.push(ratio);
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = (sorted[7] + sorted[8]) / 2.0;
    let worst = sorted.last().copied().unwrap();
    eprintln!("SWEEP: median={median:.3} worst={worst:.3} n_bad(>1.25)={n_bad}/16");
    assert!(worst.is_finite(), "ratios finite");
}

/// #1266 select-out probe: y = f(x1) only; x2 is a pure-noise irrelevant
/// covariate (independent of y). A healthy double-penalty fit must SELECT x2
/// OUT — its per-term EDF should be small (≈1, intercept-ish), i.e. ρ_null2
/// driven large. A degeneracy prior centered at ρ=0 that is too STRONG keeps
/// x2's null-space (linear) component, inflating its EDF (the #1266 regression
/// risk of over-strengthening the #1476 fix). Returns (edf_x2, total_edf).
fn select_out_probe(seed: u64) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nrm = Normal::new(0.0, 1.0).expect("normal");
    let mut g0 = Vec::with_capacity(N);
    let mut g1 = Vec::with_capacity(N);
    for _ in 0..N {
        g0.push(nrm.sample(&mut rng));
        g1.push(nrm.sample(&mut rng)); // x2 INDEPENDENT of x1 and y
    }
    let x1 = rank_to_unit(&g0);
    let x2 = rank_to_unit(&g1);
    // truth depends on x1 ONLY (a clear bend); x2 is irrelevant.
    let mu: Vec<f64> = (0..N).map(|i| (TAU * x1[i]).sin()).collect();
    let y: Vec<f64> = mu.iter().map(|&m| m + NOISE * nrm.sample(&mut rng)).collect();

    let ds = encode(&[("x1", &x1), ("x2", &x2), ("y", &y)]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x1) + s(x2)", &ds, &cfg).expect("select-out fit");
    let FitResult::Standard(fit) = result else {
        panic!("standard fit");
    };
    let total = fit.fit.edf_total().expect("edf");
    // per-term EDF via penalty_block_trace: edf_term = |coeff_range| − Σ tr_kk.
    // For s(x1)+s(x2) dp=True the blocks are [bend1, null1, bend2, null2]; x2
    // owns the trailing two. The two smooths split the columns ~evenly, so
    // x2's coeff count ≈ (p_total − 1)/2. We report total EDF and the trailing
    // block traces directly (a small trailing-trace sum = x2 selected out).
    let ll: Vec<f64> = fit.fit.log_lambdas.iter().copied().collect();
    (ll[ll.len() - 1], total)
}

#[test]
fn dp_diag_select_out_probe() {
    init_parallelism();
    for seed in [2_000_001u64, 2_000_002, 2_000_003] {
        let (rho_null2, total) = select_out_probe(seed);
        eprintln!(
            "select-out seed {seed}: rho_null2(x2)={rho_null2:.2} total_edf={total:.3} \
             (x2 is pure noise → ρ_null2 should be LARGE = selected out; total ≈ x1-only edf)"
        );
        assert!(total.is_finite(), "total edf finite");
    }
}

#[test]
fn dp_diag_single_trace_seed021() {
    init_log();
    let seed = 1_476_021u64;
    let (x1, x2, y, mu) = make_data(seed);
    eprintln!("=== single dp=true trace seed {seed} ===");
    let (err_dp, edf_dp) = fit_and_score_diag(&x1, &x2, &y, &mu, true, true);
    eprintln!("  dp rmse={err_dp:.5} edf={edf_dp:.2}");
    assert!(edf_dp.is_finite());
}

#[test]
fn dp_diag_bad_seeds_log_lambdas() {
    init_parallelism();
    init_log();
    for seed in [1_476_020u64, 1_476_021] {
        eprintln!("=== seed {seed} ===");
        let (x1, x2, y, mu) = make_data(seed);
        let (err_dp, edf_dp) = fit_and_score_diag(&x1, &x2, &y, &mu, true, true);
        let (err_sp, edf_sp) = fit_and_score_diag(&x1, &x2, &y, &mu, false, true);
        eprintln!(
            "  dp rmse={err_dp:.5} edf={edf_dp:.2} | sp rmse={err_sp:.5} edf={edf_sp:.2} | ratio={:.3}",
            err_dp / err_sp.max(1e-12)
        );
        assert!(edf_dp.is_finite() && edf_sp.is_finite(), "edf finite");
    }
}
