//! BUG-HUNT (#1074 matern_varying_nu ν=1.5): sweep explicit length_scale for the
//! ν=3/2 Matérn on the EXACT data of the failing quality test, printing the
//! interior/edge RMSE split — to see whether a shorter kernel range recovers the
//! boundaries (κ-optimization-too-long) or no range does (basis/boundary defect).

use gam::smooth::{SmoothBasisSpec, build_term_collection_design};
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

fn main() {
    init_parallelism();
    let n = 160usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, 0.08).unwrap();
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let truth = |t: f64| 0.5 + (3.0 * PI * t).sin() * (-t * t / 2.0).exp();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();
    let edge = grid_n / 10;

    // Dump the exact (x,y) data and the (x_grid, truth_grid) so an external R
    // run can fit mgcv on the identical data and report its converged range/edf.
    {
        use std::io::Write;
        let mut f = std::fs::File::create("/tmp/matern_nu_data.csv").unwrap();
        writeln!(f, "x,y").unwrap();
        for i in 0..n {
            writeln!(f, "{},{}", x[i], y[i]).unwrap();
        }
        let mut g = std::fs::File::create("/tmp/matern_nu_grid.csv").unwrap();
        writeln!(g, "xg,truth").unwrap();
        for i in 0..grid_n {
            writeln!(g, "{},{}", x_grid[i], truth_grid[i]).unwrap();
        }
    }

    let fit_grid = |formula: &str| -> Option<(Vec<f64>, f64, Option<f64>, Vec<f64>, f64)> {
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = match fit_from_formula(formula, &ds, &cfg) {
            Ok(r) => r,
            Err(e) => {
                println!("   [fit error for {formula}]: {e:?}");
                return None;
            }
        };
        let FitResult::Standard(fit) = result else {
            return None;
        };
        let mut g = Array2::<f64>::zeros((grid_n, 2));
        for (i, &t) in x_grid.iter().enumerate() {
            g[[i, 0]] = t;
        }
        let gd = build_term_collection_design(g.view(), &fit.resolvedspec).expect("grid");
        let gam_grid: Vec<f64> = gd.design.apply(&fit.fit.beta).to_vec();
        let ls = fit
            .resolvedspec
            .smooth_terms
            .iter()
            .find_map(|t| match &t.basis {
                SmoothBasisSpec::Matern { spec, .. } => Some(spec.length_scale),
                _ => None,
            });
        let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
        Some((
            gam_grid,
            edf,
            ls,
            fit.fit.log_lambdas.to_vec(),
            fit.fit.reml_score,
        ))
    };

    let report = |tag: &str, r: &Option<(Vec<f64>, f64, Option<f64>, Vec<f64>, f64)>| {
        let Some((gam_grid, edf, ls, ll, reml)) = r else {
            println!("{tag:30} FAILED");
            return;
        };
        let r_all = rmse(gam_grid, &truth_grid);
        let r_edge = rmse(
            &[&gam_grid[..edge], &gam_grid[grid_n - edge..]].concat(),
            &[&truth_grid[..edge], &truth_grid[grid_n - edge..]].concat(),
        );
        let r_int = rmse(
            &gam_grid[edge..grid_n - edge],
            &truth_grid[edge..grid_n - edge],
        );
        println!(
            "{tag:30} reml={reml:.3} rmse_all={r_all:.4} rmse_int={r_int:.4} rmse_edge={r_edge:.4} edf={edf:.2} ls={ls:?} ll={ll:?}"
        );
    };

    // Auto-optimized κ (the failing path):
    report("nu=1.5 AUTO", &fit_grid("y ~ matern(x, nu=1.5, k=18)"));
    // Explicit length-scale seeds:
    for ls in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5] {
        report(
            &format!("nu=1.5 seed_ls={ls}"),
            &fit_grid(&format!("y ~ matern(x, nu=1.5, k=18, length_scale={ls})")),
        );
    }
    report("nu=2.5 AUTO", &fit_grid("y ~ matern(x, nu=2.5, k=18)"));
}
