//! Owed-work regression gate for the #1082/#1373 REML λ-calibration cluster.
//!
//! These tests assert OBJECTIVE truth recovery (gam fit vs the known synthetic
//! surface), R-free — no mgcv/VGAM subprocess. The mature-tool comparison lives
//! in the `quality/` suite; here we pin the gam-vs-TRUTH contract so a
//! regression of the λ-selection fix fails CI without needing R installed.
//!
//! Issue: gam's production REML must not over-smooth the Poisson
//! tensor-product te() (selected λ too large → effective df too low → the
//! fitted log-mean surface biased toward flat).
//!
//! RE-DERIVED 2026-09-04 (#2668 group "owed_1082"). Two things were wrong with
//! the previous version of this gate, and neither was the engine:
//!
//! 1. The fixture's LCG took `state >> 33` — a 31-bit value — while
//!    `next_unit` divided by 2^32, so every uniform draw lived on (0, 0.5]
//!    (mean 0.249, max 0.5000) and Knuth's Poisson sampler ran at roughly half
//!    the intended rate: the counts averaged 1.08 against a "truth" surface
//!    averaging 2.45. The Poisson fit's mean equalled the data's mean exactly,
//!    so the reported "RMSE 1.41 to truth" was a correct fit of the wrong data.
//!    (`large_scale_accuracy_sweep.rs` had already found and fixed this, #1263.)
//! 2. With the sampler corrected, the original surface `0.8 + 0.3·sin x +
//!    0.2·z²` puts the counts at mean 2.5 on 300 cells, where the signal
//!    (sd of μ ≈ 0.53) is a third of the Poisson noise (sd ≈ 1.6). There the
//!    REML optimum of gam AND of mgcv sits next to the null model: gam edf 5.85,
//!    RMSE-to-truth 0.394; mgcv `te(bs="cr")` 6.35 / 0.400, `bs="bs"` 6.13 /
//!    0.399, ML 6.31 / 0.400 — all above the old `0.18·range = 0.356` bar, with
//!    58–64 % of cells inside the nominal 95 % band on both sides. The "mgcv
//!    recovers it at 10.83 edf" the bar was calibrated on is not what mgcv does
//!    on this data; at that flexibility mgcv's own scan shows RMSE 0.463.
//!
//! The fixture now uses the same surface shape at mean count ≈ 18
//! (`2.8 + 0.3·sin x + 0.2·z²`), where the surface is genuinely recoverable
//! (null-model RMSE 3.90; gam edf 17.2 / RMSE 0.672; mgcv `cr` 17.4 / 0.681,
//! `bs` 16.3 / 0.686), and asserts calibrated recovery rather than a number:
//! the fit's error at every cell is judged against the fit's own posterior
//! standard deviation. An over-smoothed fit has a small posterior and a large
//! bias, so it fails these gates by orders of magnitude; a fit that recovers
//! the surface passes them with its own uncertainty.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use gam::linalg::matrix::DesignMatrix;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Deterministic LCG + count sampler so the data are reproducible without an
/// external RNG crate dependency drift.
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
        // The top 32 bits of the 64-bit state. `>> 33` yields a 31-bit value,
        // which caps the unit draw at 0.5 and turns every sampler built on it
        // into a (0, 0.5] "uniform" — Poisson counts at roughly half the
        // intended rate (#1263; the same defect `large_scale_accuracy_sweep.rs`
        // already fixed).
        (self.state >> 32) as u32
    }
    /// Uniform on (0, 1].
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

/// The #1373 fixture surface shape `0.3·sin(x) + 0.2·z²` on a 15×20 grid,
/// x∈[0,2π], z∈[-1,1], at an intercept that puts the counts at mean ≈ 18 so
/// the surface is recoverable from Poisson noise (see the module doc);
/// counts ~ Poisson(exp(eta_true)).
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
            let eta = 2.8 + 0.3 * xi.sin() + 0.2 * zi * zi;
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
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode xzy dataset")
}

/// The fitted mean at every training cell together with its posterior
/// standard deviation on the response scale (delta method through the log
/// link: `sd(μ̂) = μ̂ · sd(η̂)`, `sd(η̂)² = xᵢᵀ Cov xᵢ`), plus the total edf.
struct TeFit {
    edf: f64,
    mean: Vec<f64>,
    sd: Vec<f64>,
    log_lambdas: Vec<f64>,
}

fn fit_te(family: &str, x: &[f64], z: &[f64], y: &[f64]) -> TeFit {
    let ds = encode_xzy(x, z, y);
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard GAM fit for {family} + te()");
    };
    let design: Array2<f64> = match &fit.design.design {
        DesignMatrix::Dense(m) => m.to_dense(),
        _ => panic!("expected a dense te() design"),
    };
    let eta = design.dot(&fit.fit.beta);
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("a default fit publishes its conditional coefficient covariance");
    let mut mean = Vec::with_capacity(eta.len());
    let mut sd = Vec::with_capacity(eta.len());
    for i in 0..eta.len() {
        let xi = design.row(i).to_owned();
        let sd_eta = xi.dot(&cov.dot(&xi)).sqrt();
        let mu = eta[i].exp();
        mean.push(mu);
        sd.push(mu * sd_eta);
    }
    TeFit {
        edf: fit.fit.edf_total().expect("edf_total"),
        mean,
        sd,
        log_lambdas: fit.fit.log_lambdas.to_vec(),
    }
}

/// #1373: gam's Poisson tensor-product te() must RECOVER the true mean surface
/// — its REML λ̂ must not over-smooth. Judged against the fit's own posterior:
///
/// * coverage: the nominal 95 % posterior band `μ̂ ± 1.96·sd(μ̂)` contains the
///   truth on at least `0.95·n − 3·√(0.95·0.05·n)` cells (the binomial count at
///   nominal coverage minus three standard deviations; Nychka's across-the-
///   function coverage is what a smoothing posterior promises);
/// * scale: the root-mean-square standardised error `(μ̂ − μ)/sd(μ̂)` over the
///   cells is at most 2 — the fit's error is within twice its own reported
///   uncertainty on average.
///
/// An over-smoothed fit (the #1373 regression: λ̂ too large, edf collapsing
/// toward the {1,x}⊗{1,z} null) has a small posterior and a large bias, and
/// fails both by orders of magnitude; the recovered surface passes with the
/// margin its own uncertainty gives it (measured: 300/300 cells covered,
/// RMS standardised error 0.61; mgcv on the same cells 0.62–0.67).
#[test]
fn poisson_tensor_te_recovers_true_mean_surface_not_oversmoothed_1373() {
    init_parallelism();
    let (x, y, z, mu_true) = poisson_tensor_grid(345);
    let n = x.len();
    assert_eq!(n, 300, "15x20 grid");
    let fit = fit_te("poisson", &x, &z, &y);
    let gam_err = rmse(&fit.mean, &mu_true);
    let ybar = y.iter().sum::<f64>() / n as f64;
    let null_err = rmse(&vec![ybar; n], &mu_true);
    let mut covered = 0usize;
    let mut sum_z2 = 0.0_f64;
    for i in 0..n {
        let zi = (fit.mean[i] - mu_true[i]) / fit.sd[i];
        sum_z2 += zi * zi;
        if zi.abs() <= 1.96 {
            covered += 1;
        }
    }
    let rms_z = (sum_z2 / n as f64).sqrt();
    let nominal = 0.95 * n as f64;
    let coverage_floor = (nominal - 3.0 * (0.95 * 0.05 * n as f64).sqrt()).floor() as usize;
    eprintln!(
        "poisson te(x,z) truth recovery (R-free): n={n} mean(y)={ybar:.3} gam_edf={:.3} \
         log_lambdas={:?} gam_rmse_to_truth={gam_err:.4} null_rmse_to_truth={null_err:.4} \
         covered={covered}/{n} (floor {coverage_floor}) rms_standardised_error={rms_z:.3}",
        fit.edf, fit.log_lambdas
    );
    assert!(
        covered >= coverage_floor,
        "Poisson te() does not recover the true mean surface: the fit's nominal 95% band \
         covers the truth on {covered}/{n} cells (floor {coverage_floor}); gam_edf={:.3}, \
         RMSE(gam, truth)={gam_err:.4} against the null model's {null_err:.4} — an \
         over-smoothed fit (λ̂ too large) reports a small posterior around a biased surface",
        fit.edf
    );
    assert!(
        rms_z <= 2.0,
        "Poisson te() error is not within its own uncertainty: RMS standardised error \
         {rms_z:.3} > 2 (gam_edf={:.3}, RMSE(gam, truth)={gam_err:.4}, null {null_err:.4})",
        fit.edf
    );
}
