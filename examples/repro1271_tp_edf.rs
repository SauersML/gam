//! Fast iteration harness for #1271 (NOT a test). Reproduces the tp single-penalty
//! EDF over-count on purely linear data and dumps EDF, lambdas so we can see that
//! REML under-penalizes wiggle. DGP: y = 2 + 3x + N(0, 0.15), x = linspace(0,1,800),
//! k=20, Gaussian, REML. mgcv gives EDF ~= 2.10; gam reportedly ~4.87.
//!
//! Run: `cargo run --profile release-dev --example repro1271_tp_edf`

use gam::{FitConfig, FitResult, fit_from_formula, load_csvwith_inferred_schema};
use std::io::Write;
use std::path::PathBuf;

fn write_linear_csv(n: usize, seed: u64) -> PathBuf {
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut normal = || {
        let u1 = next_unit().max(1e-12);
        let u2 = next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let path = std::env::temp_dir().join(format!("repro1271_seed{seed}.csv"));
    let mut f = std::fs::File::create(&path).expect("create csv");
    writeln!(f, "x,y").unwrap();
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        let y = 2.0 + 3.0 * x + 0.15 * normal();
        writeln!(f, "{x:.10},{y:.10}").unwrap();
    }
    path
}

fn main() {
    gam::init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula =
        std::env::var("REPRO_FORMULA").unwrap_or_else(|_| "y ~ s(x, bs=\"tp\", k=20)".to_string());
    println!("formula: {formula}");
    let mut edfs = Vec::new();
    for seed in [1u64, 2, 3, 4, 5] {
        let path = write_linear_csv(800, seed);
        let ds = load_csvwith_inferred_schema(&path).expect("load csv");
        let result = fit_from_formula(&formula, &ds, &cfg).expect("gam tp fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected standard fit");
        };
        let inf = fit.fit.inference.as_ref().expect("inference present");
        edfs.push(inf.edf_total);
        println!(
            "seed={seed} edf_total={:.4} edf_by_block={:?} lambdas={:?}",
            inf.edf_total,
            inf.edf_by_block,
            fit.fit.lambdas.to_vec(),
        );
        if seed == 1 {
            // Penalty spectrum (the diagonal radial-eigenvalue penalty for tp).
            for bp in &fit.design.penalties {
                let s = &bp.local;
                let (rows, cols) = s.dim();
                if rows == cols && rows > 1 {
                    let mut diag: Vec<f64> = (0..rows).map(|i| s[[i, i]]).collect();
                    diag.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    println!("  [spectrum] block {:?} dim={rows} diag(desc)={:?}", bp.col_range,
                        diag.iter().map(|v| format!("{v:.4e}")).collect::<Vec<_>>());
                }
            }
            // Per-coefficient EDF = diag of the influence (hat) matrix in coef space.
            if let Some(infl) = inf.coefficient_influence.as_ref() {
                let p = infl.nrows().min(infl.ncols());
                let per: Vec<f64> = (0..p).map(|i| infl[[i, i]]).collect();
                let total: f64 = per.iter().sum();
                println!("  [per-coef-edf] sum={total:.4} diag={:?}",
                    per.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
            }
        }
    }
    let mean = edfs.iter().sum::<f64>() / edfs.len() as f64;
    println!("EDF mean = {mean:.4}  (mgcv ~= 2.10)");
}
