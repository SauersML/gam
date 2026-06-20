//! BUG-HUNT (3-D tensor te(x,y,z) recovery): a smooth additive-plus-interaction
//! 3-D surface
//!     f(x,y,z) = sin(2x) + (y-0.5)^2 + 0.5*x*z,   x,y,z ∈ [0,1]
//! fit with te(x,y,z,k=[5,5,5]) Gaussian. Checks that the 3-D Kronecker tensor
//! construction (125 coeffs, 3 marginal penalties) actually RECOVERS the surface
//! out-of-sample rather than blowing up conditioning or collapsing to a constant.
//!
//! Oracle: a 3-D thin-plate s(x,y,z,bs="tp") on the same data should recover the
//! same surface; te() held-out RMSE must be in the same ballpark (not 2x+ worse),
//! and must beat the trivial constant predictor by a wide margin.
//! Multi-seed.

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

fn truth(x: f64, y: f64, z: f64) -> f64 {
    (2.0 * x).sin() + (y - 0.5) * (y - 0.5) + 0.5 * x * z
}

fn fit_score(
    formula: &str,
    xtr: &[f64],
    ytr_v: &[f64],
    ztr: &[f64],
    resp: &[f64],
    xte: &[f64],
    yte: &[f64],
    zte: &[f64],
) -> (f64, f64) {
    let data = encode_columns(&["x", "y", "z", "r"], &[xtr, ytr_v, ztr, resp]);
    let col = data.column_map();
    let (xi, yi, zi) = (col["x"], col["y"], col["z"]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula(formula, &data, &cfg).expect("3d fit");
    let FitResult::Standard(fit) = res else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    let nte = xte.len();
    let mut grid = Array2::<f64>::zeros((nte, data.headers.len()));
    for i in 0..nte {
        grid[[i, xi]] = xte[i];
        grid[[i, yi]] = yte[i];
        grid[[i, zi]] = zte[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let eta = design.design.apply(&fit.fit.beta);
    let mut tr: Vec<f64> = (0..nte).map(|i| truth(xte[i], yte[i], zte[i])).collect();
    let mh: f64 = eta.iter().sum::<f64>() / nte as f64;
    let mt: f64 = tr.iter().sum::<f64>() / nte as f64;
    for v in tr.iter_mut() {
        *v -= mt;
    }
    let mut sse = 0.0;
    for i in 0..nte {
        let d = (eta[i] - mh) - tr[i];
        sse += d * d;
    }
    (edf, (sse / nte as f64).sqrt())
}

fn main() {
    init_parallelism();
    // Random-ish space-filling sample (Halton-like via LCG) rather than a full
    // 3-D grid to keep n moderate.
    let n = 600usize;
    let nte = 400usize;
    let mut samp = |seed: u64, m: usize| {
        let mut rng = Lcg::new(seed);
        let mut x = vec![];
        let mut y = vec![];
        let mut z = vec![];
        for _ in 0..m {
            x.push(rng.next_unit());
            y.push(rng.next_unit());
            z.push(rng.next_unit());
        }
        (x, y, z)
    };
    let (xte, yte, zte) = samp(999, nte);
    // Constant-predictor baseline RMSE (truth std around its mean on the test set).
    let tvals: Vec<f64> = (0..nte).map(|i| truth(xte[i], yte[i], zte[i])).collect();
    let mt: f64 = tvals.iter().sum::<f64>() / nte as f64;
    let const_rmse =
        (tvals.iter().map(|v| (v - mt) * (v - mt)).sum::<f64>() / nte as f64).sqrt();
    println!("3-D te(x,y,z,k=[5,5,5]) recovery; n_train={n}, n_test={nte}, const-predictor RMSE={const_rmse:.4}");
    println!("seed | te edf  heldRMSE | tp edf  heldRMSE | te/tp ratio");
    let noise = 0.15;
    let seeds = 4u64;
    let (mut ter_s, mut tpr_s) = (0.0, 0.0);
    for seed in 1u64..=seeds {
        let (xtr, ytr_v, ztr) = samp(seed.wrapping_mul(2654435761), n);
        let mut rng = Lcg::new(seed.wrapping_add(7));
        let resp: Vec<f64> = (0..n)
            .map(|i| truth(xtr[i], ytr_v[i], ztr[i]) + noise * rng.next_normal())
            .collect();
        let (tee, ter) =
            fit_score("r ~ te(x, y, z, k=[5,5,5])", &xtr, &ytr_v, &ztr, &resp, &xte, &yte, &zte);
        let (tpe, tpr) =
            fit_score("r ~ s(x, y, z, bs='tp')", &xtr, &ytr_v, &ztr, &resp, &xte, &yte, &zte);
        ter_s += ter;
        tpr_s += tpr;
        println!(
            "{seed:>4} | {tee:6.2} {ter:8.4} | {tpe:6.2} {tpr:8.4} | {:.2}x",
            ter / tpr
        );
    }
    let f = seeds as f64;
    println!(
        "MEAN | te heldRMSE={:.4} | tp heldRMSE={:.4} | te/tp={:.2}x | te/const={:.2}x",
        ter_s / f,
        tpr_s / f,
        ter_s / tpr_s,
        (ter_s / f) / const_rmse
    );
    println!("(Correct: te heldRMSE << const RMSE, and te/tp ratio ~1 (not 2x+ worse). te/const ~1 => 3-D te collapsed.)");
}
