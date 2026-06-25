//! BUG-HUNT (#1074 spatial under-recovery): reproduce the EXACT data of the
//! matern and duchon truth-recovery quality tests (same StdRng seeds) and print
//! gam's converged fit diagnostics — RMSE-vs-truth on the test grid, total EDF,
//! and the converged smoothing λ / κ — WITHOUT needing R on the node.
//!
//! The mgcv baselines (measured locally, mgcv 1.9.4, identical data) are:
//!   * matern s(x,bs="gp",m=4) k=20 : rmse_vs_truth = 0.0308, edf = 13.11
//!   * duchon s(x,z,bs="ds",m=c(2,0)) k=49 : rmse_vs_truth = 0.0233, edf = 38.70
//! The quality tests assert gam recovers within 1.10x of mgcv. This probe prints
//! gam's numbers so we can see WHETHER and BY HOW MUCH gam under-recovers, and
//! whether it is over-smoothing (low EDF) — the #1074 capped-prior hypothesis.

use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    matrix::LinearOperator,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n as f64).sqrt()
}

fn matern_probe() {
    // EXACT reproduction of quality_vs_mgcv_matern_smooth::gam_matern_smooth_recovers_truth
    let n = 180usize;
    let mut rng = StdRng::seed_from_u64(456);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, 0.08).unwrap();
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let truth = |t: f64| 1.0 + 0.8 * (4.0 * PI * t).sin() + 0.4 * (2.0 * PI * t).cos();
    let y: Vec<f64> = x.iter().map(|&t| truth(t) + noise.sample(&mut rng)).collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=2.5, k=20)", &ds, &cfg).expect("gam matern fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    let grid_n = n;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
    }
    let grid_design =
        build_term_collection_design(g.view(), &fit.resolvedspec).expect("rebuild grid");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();

    let r = rmse(&gam_grid, &truth_grid);
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    println!(
        "MATERN  gam rmse_vs_truth={r:.4}  edf={edf:.2}  lambdas={:?}",
        fit.fit.lambdas
    );
    println!("        mgcv baseline: rmse=0.0308 edf=13.11 | bars: rmse<0.08 AND rmse<=0.0339");
}

fn duchon_probe() {
    // EXACT reproduction of quality_vs_mgcv_duchon_2d::gam_duchon_2d_surface_matches_mgcv_ds
    fn truth_surface(x: f64, z: f64) -> f64 {
        let bump = |cx: f64, cz: f64, s: f64, a: f64| {
            let d2 = (x - cx).powi(2) + (z - cz).powi(2);
            a * (-d2 / (2.0 * s * s)).exp()
        };
        bump(0.3, 0.3, 0.18, 1.0) + bump(0.7, 0.65, 0.22, 0.8)
    }
    let n = 400usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).unwrap();
    let noise = Normal::new(0.0, 0.10).unwrap();
    let (mut x, mut z, mut y) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(truth_surface(xi, zi) + noise.sample(&mut rng));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode duchon");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, z, k=49)", &ds, &cfg).expect("gam duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    let gsz = 25usize;
    let coord = |i: usize| 0.05 + 0.90 * i as f64 / (gsz as f64 - 1.0);
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);
    let mut gx = vec![];
    let mut gz = vec![];
    let mut y_truth = vec![];
    for i in 0..gsz {
        for j in 0..gsz {
            let (xi, zi) = (coord(i), coord(j));
            gx.push(xi);
            gz.push(zi);
            y_truth.push(truth_surface(xi, zi));
        }
    }
    let m = gx.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = gx[i];
        grid[[i, z_idx]] = gz[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    let r = rmse(&gam_fitted, &y_truth);
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    println!(
        "DUCHON  gam rmse_vs_truth={r:.4}  edf={edf:.2}  lambdas={:?}",
        fit.fit.lambdas
    );
    println!("        mgcv baseline: rmse=0.0233 edf=38.70 | bars: rmse<0.15 AND rmse<=0.0256");
}

fn main() {
    init_parallelism();
    matern_probe();
    duchon_probe();
}
