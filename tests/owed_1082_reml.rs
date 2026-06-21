//! Owed-work regression gate for the #1082/#1373 REML λ-calibration cluster.
//!
//! These tests assert OBJECTIVE truth recovery (gam fit vs the known synthetic
//! surface), R-free — no mgcv/VGAM subprocess. The mature-tool comparison lives
//! in the `quality/` suite; here we pin the gam-vs-TRUTH contract so a
//! regression of the λ-selection fix fails CI without needing R installed.
//!
//! Issue: gam's production REML over-smooths the Poisson tensor-product te()
//! (selected λ too large → effective df too low → the fitted log-mean surface is
//! biased toward flat), so the held-out recovery of the true mean surface is
//! worse than the irreducible-noise bar. The fix must let λ̂ reach the genuine
//! REML optimum so gam recovers the surface.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Deterministic LCG + Box-Muller / count sampler so the data are reproducible
/// without an external RNG crate dependency drift. Same shape as the bug-hunt
/// probe `examples/probe_poisson_tensor_oversmooth.rs`.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
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
    /// Knuth's multiplicative Poisson sampler (exact for the small rates here).
    fn poisson(&mut self, lam: f64) -> f64 {
        let l = (-lam).exp();
        let mut k = 0u32;
        let mut p = 1.0_f64;
        loop {
            p *= self.next_unit();
            if p <= l {
                break;
            }
            k += 1;
            if k > 10_000 {
                break;
            }
        }
        k as f64
    }
}

/// The exact #1373 fixture surface: `eta_true = 0.8 + 0.3·sin(x) + 0.2·z²` on a
/// 15×20 grid, x∈[0,2π], z∈[-1,1]; counts ~ Poisson(exp(eta_true)).
fn poisson_tensor_grid(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = 15usize;
    let nz = 20usize;
    let mut rng = Lcg::new(seed.wrapping_mul(2654435761));
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut y = Vec::with_capacity(nx * nz);
    let mut mu_true = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        let xi = (ix as f64) / ((nx - 1) as f64) * (2.0 * PI);
        for iz in 0..nz {
            let zi = -1.0 + 2.0 * (iz as f64) / ((nz - 1) as f64);
            let eta = 0.8 + 0.3 * xi.sin() + 0.2 * zi * zi;
            let mu = eta.exp();
            x.push(xi);
            z.push(zi);
            y.push(rng.poisson(mu));
            mu_true.push(mu);
        }
    }
    (x, y, z, mu_true)
}

fn encode_xzy(x: &[f64], z: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..x.len())
        .map(|i| {
            StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode xzy dataset")
}

/// Fit `y ~ te(x, z, k=[6,6])` for `family`; return (edf_total, fitted mean on
/// the response scale at the training points).
fn fit_te_mean(family: &str, x: &[f64], z: &[f64], y: &[f64]) -> (f64, Vec<f64>) {
    let ds = encode_xzy(x, z, y);
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard GAM fit for {family} + te()");
    };
    let edf = fit.fit.edf_total().expect("edf_total");
    let n = x.len();
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild te design");
    let eta = design.design.apply(&fit.fit.beta);
    let mean: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    (edf, mean)
}

/// #1373: gam's Poisson tensor-product te() must RECOVER the true mean surface,
/// i.e. its REML λ̂ must not over-smooth. The bar is the same absolute
/// truth-recovery bound the mgcv quality test uses (0.18·range), R-free.
///
/// This is the regression gate: before the λ-calibration fix gam reports
/// edf≈6.9 and RMSE-to-truth ≈ 2.2× the mgcv error, blowing the bar; after the
/// fix gam reaches edf≈10 and recovers the surface within the bound.
#[test]
fn poisson_tensor_te_recovers_true_mean_surface_not_oversmoothed_1373() {
    init_parallelism();
    let (x, y, z, mu_true) = poisson_tensor_grid(345);
    let n = x.len();
    assert_eq!(n, 300, "15x20 grid");

    let (gam_edf, gam_mean) = fit_te_mean("poisson", &x, &z, &y);
    let gam_err = rmse(&gam_mean, &mu_true);

    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = mu_max - mu_min;
    let abs_bar = 0.18 * mu_range;

    eprintln!(
        "poisson te(x,z) truth recovery (R-free): n={n} mu_range={mu_range:.4} \
         gam_edf={gam_edf:.3} gam_rmse_to_truth={gam_err:.4} abs_bar={abs_bar:.4}"
    );

    // PRIMARY: recover the true mean surface within the irreducible-noise bar.
    assert!(
        gam_err <= abs_bar,
        "Poisson te() over-smoothed: RMSE(gam, truth)={gam_err:.4} > {abs_bar:.4} \
         (0.18·range); gam_edf={gam_edf:.3} (under-flexible — the true surface needs \
         ~10 edf, mgcv recovers it at 10.83)"
    );

    // SECONDARY: the effective df must reach the flexibility the smooth surface
    // genuinely needs. The truth sin(x)+z² over k=[6,6] supports ~10 edf; an
    // over-smoothed fit collapses toward the {1,x}⊗{1,z} ~4-edf null. A floor of
    // 8.5 is comfortably below mgcv's 10.83 and well above the over-smoothed 6.9,
    // so it fails on the regression and passes on the fix without being brittle.
    assert!(
        gam_edf >= 8.5,
        "Poisson te() effective df {gam_edf:.3} too low (over-smoothed); the true \
         surface needs ~10 edf (mgcv 10.83). A REML λ over-selection regressed."
    );
    assert!(
        gam_edf < 30.0,
        "Poisson te() effective df {gam_edf:.3} implausibly high (under-smoothed)"
    );
}
