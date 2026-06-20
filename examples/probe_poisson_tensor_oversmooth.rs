//! BUG-HUNT (#1373 poisson_tensor over-smooth): the failing fixture
//! `quality_vs_mgcv_poisson_tensor` reports gam_edf=6.90 vs mgcv 10.83 on a
//! te(x,z,k=[6,6]) Poisson(log) fit of a smooth surface
//!     eta_true = 0.8 + 0.3·sin(x) + 0.2·z²,  x∈[0,2π], z∈[-1,1], n=300.
//! gam is UNDER-FLEXIBLE (over-smoothing) → 2.18× worse RMSE-to-truth than mgcv.
//!
//! This probe isolates WHERE the over-smoothing comes from by fitting the SAME
//! tensor te(x,z,k=[6,6]) on the SAME (x,z) grid two ways:
//!   - POISSON(log) on the actual count draws  (the failing path)
//!   - GAUSSIAN(identity) on eta_true + matched Gaussian noise  (control)
//! and reporting, per family and per seed:
//!   - edf_total
//!   - RMSE of fitted eta vs eta_true  (recovery on the linear-predictor scale)
//!
//! If the Gaussian tensor recovers the surface (high EDF, low RMSE) while the
//! Poisson tensor collapses to low EDF / high RMSE on the SAME structure, the
//! over-smoothing lives in the GLM-family REML λ-selection path (working-weight
//! → LAML λ-choice), not in the tensor construction (which is family-agnostic).
//! Multi-seed to rule out a single unlucky draw.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    matrix::LinearOperator,
};
use ndarray::Array2;
use std::f64::consts::PI;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / ((u32::MAX as f64) + 1.0)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
    fn poisson(&mut self, lam: f64) -> f64 {
        let mut k = 0u32;
        let mut s = 0.0_f64;
        let limit = lam.max(1.0) * 60.0 + 60.0;
        loop {
            s += -self.next_unit().ln();
            if s > lam {
                break;
            }
            k += 1;
            if k as f64 > limit {
                break;
            }
        }
        k as f64
    }
}

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

/// Fit `y ~ te(x, z, k=[6,6])` for the given family; return (edf, eta_rmse_vs_truth).
fn fit_te(family: &str, x: &[f64], z: &[f64], y: &[f64], eta_true: &[f64]) -> (f64, f64, Vec<f64>) {
    let data = encode_columns(&["x", "z", "y"], &[x, z, y]);
    let col = data.column_map();
    let xi = col["x"];
    let zi = col["z"];
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula("y ~ te(x, z, k=[6,6])", &data, &cfg).expect("te fit");
    let FitResult::Standard(fit) = res else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    let lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    let n = x.len();
    let mut grid = Array2::<f64>::zeros((n, data.headers.len()));
    for i in 0..n {
        grid[[i, xi]] = x[i];
        grid[[i, zi]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let eta_hat = design.design.apply(&fit.fit.beta);
    // shape RMSE: remove common mean offset.
    let mh: f64 = eta_hat.iter().sum::<f64>() / n as f64;
    let mt: f64 = eta_true.iter().sum::<f64>() / n as f64;
    let mut sse = 0.0;
    for i in 0..n {
        let d = (eta_hat[i] - mh) - (eta_true[i] - mt);
        sse += d * d;
    }
    (edf, (sse / n as f64).sqrt(), lambdas)
}

fn fmt_lams(l: &[f64]) -> String {
    l.iter()
        .map(|v| format!("{v:.2e}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    init_parallelism();
    let nx = 15usize;
    let nz = 20usize;
    // Build the fixed (x,z) grid and eta_true exactly as the failing fixture.
    let mut x = vec![];
    let mut z = vec![];
    let mut eta_true = vec![];
    for ix in 0..nx {
        let xi = (ix as f64) / ((nx - 1) as f64) * (2.0 * PI);
        for iz in 0..nz {
            let zi = -1.0 + 2.0 * (iz as f64) / ((nz - 1) as f64);
            x.push(xi);
            z.push(zi);
            eta_true.push(0.8 + 0.3 * xi.sin() + 0.2 * zi * zi);
        }
    }
    let n = x.len();
    // Match the Gaussian noise scale to the Poisson signal-to-noise on the eta
    // scale: Var(Poisson count)=mu, so on the log/eta scale the delta-method sd
    // is ~1/sqrt(mu). Use the mean 1/sqrt(mu) as the Gaussian eta-noise sd so the
    // two families face comparable difficulty recovering eta_true.
    let mean_inv_sqrt_mu: f64 =
        eta_true.iter().map(|&e| 1.0 / e.exp().sqrt()).sum::<f64>() / n as f64;
    println!("n={n}  matched Gaussian eta-noise sd ~ {mean_inv_sqrt_mu:.3}");
    println!(
        "seed | GAUSS edf  rmse | POIS edf  rmse   (truth needs a few df; fixture mgcv edf=10.83, gam=6.90)"
    );
    let (mut ge_s, mut gr_s, mut pe_s, mut pr_s) = (0.0, 0.0, 0.0, 0.0);
    let seeds = 8u64;
    for seed in 1u64..=seeds {
        let mut rng = Lcg::new(seed.wrapping_mul(2654435761));
        // Poisson counts from eta_true.
        let yp: Vec<f64> = eta_true.iter().map(|&e| rng.poisson(e.exp())).collect();
        // Gaussian eta + matched noise.
        let yg: Vec<f64> = eta_true
            .iter()
            .map(|&e| e + mean_inv_sqrt_mu * rng.next_normal())
            .collect();
        let (ge, gr, gl) = fit_te("gaussian", &x, &z, &yg, &eta_true);
        let (pe, pr, pl) = fit_te("poisson", &x, &z, &yp, &eta_true);
        ge_s += ge;
        gr_s += gr;
        pe_s += pe;
        pr_s += pr;
        println!(
            "{seed:>4} | {ge:8.3} {gr:7.4} [{}] | {pe:8.3} {pr:7.4} [{}]",
            fmt_lams(&gl),
            fmt_lams(&pl)
        );
    }
    let f = seeds as f64;
    println!(
        "MEAN | GAUSS edf={:.3} rmse={:.4} | POIS edf={:.3} rmse={:.4}",
        ge_s / f,
        gr_s / f,
        pe_s / f,
        pr_s / f
    );
    println!(
        "(If POIS edf << GAUSS edf and POIS rmse >> GAUSS rmse on the SAME surface => family-path over-smoothing.)"
    );
    println!(
        "([..] = selected per-margin lambdas; larger lambda = more smoothing. Poisson lambda >> Gaussian lambda confirms family over-selection.)"
    );
}
