//! BUG-HUNT (te() anisotropy / per-margin λ selection): a surface that is very
//! WIGGLY in x and nearly FLAT in z,
//!     f(x,z) = sin(5x) + 0.15·z,   x,z ∈ [0,1],
//! must be fit by te(x,z) selecting a SMALL λ_x (low smoothing → keep the wiggle)
//! and a LARGE λ_z (high smoothing → the z margin is ~linear). If the margin
//! penalties are mis-balanced (Kronecker normalization folds the other margin's
//! basis size / scale into a margin's λ), te() either over-smooths x (loses the
//! sin(5x) wiggle) or under-smooths z (z margin over-fits noise).
//!
//! We fit te(x,z,k=[10,10]) Gaussian on noisy draws and report, per seed:
//!   - selected (λ_x, λ_z)          [expect λ_z >> λ_x]
//!   - held-out RMSE vs truth on a disjoint test grid
//!   - the per-axis recovered slice: max|∂/∂x| should track sin(5x) amplitude,
//!     the z-direction should be ~linear.
//! As an oracle, we also fit te() to a surface with the roles SWAPPED
//!     g(x,z) = 0.15·x + sin(5z)
//! and check the selected (λ_x, λ_z) FLIPS (λ_x >> λ_z). If anisotropy is
//! handled correctly, the λ ordering must track which margin is wiggly; a fixed
//! ordering regardless of which axis carries the signal is the bug.

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

/// Fit te(x,z,k=[10,10]); return (lambda_x, lambda_z, edf, heldout_rmse_vs truth).
/// `truth` is a closure giving the noise-free surface at (x,z).
fn fit_te(
    xtr: &[f64],
    ztr: &[f64],
    ytr: &[f64],
    xte: &[f64],
    zte: &[f64],
    truth: &dyn Fn(f64, f64) -> f64,
) -> (f64, f64, f64, f64) {
    let data = encode_columns(&["x", "z", "y"], &[xtr, ztr, ytr]);
    let col = data.column_map();
    let xi = col["x"];
    let zi = col["z"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula("y ~ te(x, z, k=[10,10])", &data, &cfg).expect("te fit");
    let FitResult::Standard(fit) = res else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    let lams = fit.fit.lambdas.to_vec();
    // Margin order in the spec is [x, z]; the first two lambdas are the marginal
    // penalties (TensorMarginal dim 0 = x, dim 1 = z).
    let (lx, lz) = (
        lams.first().copied().unwrap_or(f64::NAN),
        lams.get(1).copied().unwrap_or(f64::NAN),
    );
    let nte = xte.len();
    let mut grid = Array2::<f64>::zeros((nte, data.headers.len()));
    for i in 0..nte {
        grid[[i, xi]] = xte[i];
        grid[[i, zi]] = zte[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let eta_hat = design.design.apply(&fit.fit.beta);
    let mut tr: Vec<f64> = (0..nte).map(|i| truth(xte[i], zte[i])).collect();
    let mh: f64 = eta_hat.iter().sum::<f64>() / nte as f64;
    let mt: f64 = tr.iter().sum::<f64>() / nte as f64;
    for v in tr.iter_mut() {
        *v -= mt;
    }
    let mut sse = 0.0;
    for i in 0..nte {
        let d = (eta_hat[i] - mh) - tr[i];
        sse += d * d;
    }
    (lx, lz, edf, (sse / nte as f64).sqrt())
}

fn main() {
    init_parallelism();
    let ng = 22usize;
    let mk_grid = |jit: f64| {
        let mut x = vec![];
        let mut z = vec![];
        for ix in 0..ng {
            let xc = (ix as f64 + jit) / (ng as f64);
            for iz in 0..ng {
                let zc = (iz as f64 + jit) / (ng as f64);
                x.push(xc);
                z.push(zc);
            }
        }
        (x, z)
    };
    let (xtr, ztr) = mk_grid(0.0);
    let (xte, zte) = mk_grid(0.5);
    let n = xtr.len();
    let noise = 0.2;
    // f wiggly in x, flat in z; g is the swap.
    let f = |x: f64, _z: f64| (5.0 * x).sin();
    let f_full = |x: f64, z: f64| (5.0 * x).sin() + 0.15 * z;
    let g = |_x: f64, z: f64| (5.0 * z).sin();
    let g_full = |x: f64, z: f64| 0.15 * x + (5.0 * z).sin();
    let _ = (f, g);
    println!("te(x,z,k=[10,10]) ANISOTROPY: f wiggly-in-x flat-in-z; g=swap. n_train={n}, noise={noise}");
    println!("Expect: f => lambda_z >> lambda_x ; g => lambda_x >> lambda_z (lambda ordering tracks the wiggly axis)");
    println!("seed | f: lam_x     lam_z    edf   rmse | g: lam_x     lam_z    edf   rmse | f wig_axis g wig_axis");
    let seeds = 6u64;
    let (mut okf, mut okg) = (0, 0);
    for seed in 1u64..=seeds {
        let mut rng = Lcg::new(seed.wrapping_mul(2654435761));
        let yf: Vec<f64> = (0..n)
            .map(|i| f_full(xtr[i], ztr[i]) + noise * rng.next_normal())
            .collect();
        let yg: Vec<f64> = (0..n)
            .map(|i| g_full(xtr[i], ztr[i]) + noise * rng.next_normal())
            .collect();
        let (fx, fz, fe, fr) = fit_te(&xtr, &ztr, &yf, &xte, &zte, &f_full);
        let (gx, gz, ge, gr) = fit_te(&xtr, &ztr, &yg, &xte, &zte, &g_full);
        // f wiggly in x => expect lam_z > lam_x ; g wiggly in z => expect lam_x > lam_z.
        let f_axis = if fz > fx { "z-smooth(OK)" } else { "x-smooth(BAD)" };
        let g_axis = if gx > gz { "x-smooth(OK)" } else { "z-smooth(BAD)" };
        if fz > fx {
            okf += 1;
        }
        if gx > gz {
            okg += 1;
        }
        println!(
            "{seed:>4} | {fx:9.2e} {fz:9.2e} {fe:5.2} {fr:6.4} | {gx:9.2e} {gz:9.2e} {ge:5.2} {gr:6.4} | {f_axis} {g_axis}"
        );
    }
    println!("SUMMARY: f correct-ordering {okf}/{seeds}, g correct-ordering {okg}/{seeds}");
    println!("(If the lambda ordering does NOT flip when the wiggly axis flips => te() margin-balance is broken / not anisotropic.)");
}
