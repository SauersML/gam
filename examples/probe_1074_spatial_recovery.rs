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

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(s: &mut u64) -> f64 {
    (splitmix64(s) >> 11) as f64 / (1u64 << 53) as f64
}

fn tpte_probe() {
    // EXACT reproduction of quality_vs_mgcv_tensor_additive_tp_te::gam_additive_tp_plus_te_matches_mgcv
    // s(x1,x2,bs="tp",k=10) + te(z,w,k=6), noise-free additive truth, n=500.
    const N: usize = 500;
    let mut state: u64 = 20260530;
    let mut x1 = vec![];
    let mut x2 = vec![];
    let mut z = vec![];
    let mut w = vec![];
    let mut truth = vec![];
    for _ in 0..N {
        let a = next_unit(&mut state);
        let b = next_unit(&mut state);
        let c = next_unit(&mut state);
        let d = next_unit(&mut state);
        let f1 = (PI * a).sin() * (-b).exp();
        let f2 = c * c * (PI * d).cos();
        x1.push(a);
        x2.push(b);
        z.push(c);
        w.push(d);
        truth.push(f1 + f2);
    }
    let signal_range =
        truth.iter().copied().fold(f64::NEG_INFINITY, f64::max) - truth.iter().copied().fold(f64::INFINITY, f64::min);
    let headers = ["x1", "x2", "z", "w", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", w[i]),
                format!("{:.17e}", truth[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tpte");
    let col = ds.column_map();
    let (x1i, x2i, zi, wi) = (col["x1"], col["x2"], col["z"], col["w"]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, x2, bs=\"tp\", k=10) + te(z, w, k=6)", &ds, &cfg)
        .expect("gam tp+te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1i]] = x1[i];
        grid[[i, x2i]] = x2[i];
        grid[[i, zi]] = z[i];
        grid[[i, wi]] = w[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let r = rmse(&gam_fitted, &truth);
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    println!(
        "TP+TE   gam rmse_vs_truth={r:.6} ({:.3}% range)  edf={edf:.2}  lambdas={:?}",
        100.0 * r / signal_range,
        fit.fit.lambdas
    );
    println!("        mgcv baseline: rmse=0.029546 (1.247% range) edf=38.66 | bars: <2% AND <=1.37%");
}

fn sz_probe() {
    // EXACT reproduction of quality_vs_mgcv_factor_smooth_sz::gam_factor_smooth_sz_matches_mgcv
    // y ~ s(group, x, bs="sz"); 6 groups x 60 = 360 rows; f_g(x)=sin(2pi x)*z_g, sum z_g=0.
    const N_GROUPS: usize = 6;
    const PER_GROUP: usize = 60;
    const SEED: u64 = 77;
    const NOISE_SD: f64 = 0.2;
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let znorm = Normal::new(0.0, 0.8).unwrap();
    let epsd = Normal::new(0.0, 0.2).unwrap();
    let mut z: Vec<f64> = (0..N_GROUPS).map(|_| znorm.sample(&mut rng)).collect();
    let zbar: f64 = z.iter().sum::<f64>() / N_GROUPS as f64;
    for zi in z.iter_mut() {
        *zi -= zbar;
    }
    let two_pi = 2.0 * PI;
    let n = N_GROUPS * PER_GROUP;
    let mut group_code = vec![];
    let mut group_str = vec![];
    let mut x = vec![];
    let mut y = vec![];
    let mut truth = vec![];
    for g in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let fi = (two_pi * xi).sin() * z[g];
            let yi = fi + epsd.sample(&mut rng);
            group_code.push(g as f64);
            group_str.push(format!("g{g}"));
            x.push(xi);
            y.push(yi);
            truth.push(fi);
        }
    }
    let headers: Vec<String> = vec!["group".into(), "x".into(), "y".into()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![group_str[i].clone(), x[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode sz");
    let col = ds.column_map();
    let (group_idx, x_idx) = (col["group"], col["x"]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(group, x, bs=\"sz\")", &ds, &cfg).expect("gam sz fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let n_cols = ds.headers.len();
    let mut train_grid = Array2::<f64>::zeros((n, n_cols));
    for i in 0..n {
        train_grid[[i, group_idx]] = group_code[i];
        train_grid[[i, x_idx]] = x[i];
    }
    let train_design =
        build_term_collection_design(train_grid.view(), &fit.resolvedspec).expect("rebuild sz");
    let intercept = fit.fit.beta[train_design.intercept_range.start];
    let gam_fitted: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();
    let gam_term: Vec<f64> = gam_fitted.iter().map(|v| v - intercept).collect();
    let r = rmse(&gam_term, &truth);
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    println!(
        "SZ      gam rmse_vs_truth={r:.5}  edf={edf:.2}  log_lambdas={:?}",
        fit.fit.log_lambdas
    );
    println!("        noise_sd bar={NOISE_SD} ; mgcv match-or-beat 1.10x (was reported ~1.23x gap)");
}

fn main() {
    init_parallelism();
    matern_probe();
    duchon_probe();
    tpte_probe();
    sz_probe();
}
