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
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    PredictUncertaintyOptions, PredictUncertaintyResult, predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Deterministic LCG + count sampler so the data are reproducible
/// without an external RNG crate dependency drift.
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
        (self.state >> 32) as u32
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
        }
        k as f64
    }
}

#[test]
fn poisson_fixture_sampler_has_the_declared_mean_and_variance() {
    let mut rng = Lcg::new(1082);
    let n = 100_000;
    let rate = 2.5;
    let mut sum = 0.0;
    let mut squares = 0.0;
    for _ in 0..n {
        let count = rng.poisson(rate);
        sum += count;
        squares += count * count;
    }
    let mean = sum / n as f64;
    let variance = squares / n as f64 - mean * mean;
    assert!((mean - rate).abs() < 0.03, "Poisson mean: {mean}");
    assert!(
        (variance - rate).abs() < 0.06,
        "Poisson variance: {variance}"
    );
}

/// The exact #1373 fixture surface: `eta_true = 0.8 + 0.3·sin(x) + 0.2·z²` on a
/// x∈[0,2π], z∈[-1,1]; counts ~ Poisson(exp(eta_true)).
///
/// The quadratic coefficient is only 0.2. After removing the intercept,
/// Var(z²)=4/45, and the smallest mean is exp(0.5). A five-standard-error
/// signal therefore needs at least 25/(exp(0.5)*(4/45)*0.2²) ≈ 4265 rows.
/// The 75×60 grid clears that information requirement. At the old N=300,
/// even mgcv selects EDF=6.35 and misses the absolute recovery bar (RMSE=.400),
/// so the alleged EDF=10.83 reference did not describe this fixture.
fn poisson_tensor_grid(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = 75usize;
    let nz = 60usize;
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
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode xzy dataset")
}

/// Fit the Poisson tensor and predict through the production posterior API.
fn fit_te_mean(x: &[f64], z: &[f64], y: &[f64]) -> (f64, PredictUncertaintyResult) {
    let ds = encode_xzy(x, z, y);
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let cfg = FitConfig {
        family: Some("poisson".into()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard GAM fit for Poisson + te()");
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
    let offset = Array1::zeros(n);
    let prediction = predict_gamwith_uncertainty(
        design.design,
        fit.fit.beta.view(),
        offset.view(),
        LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ),
        &fit.fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("posterior-mean Poisson tensor prediction");
    (edf, prediction)
}

/// #1373: gam's Poisson tensor-product te() must RECOVER the true mean surface,
/// i.e. its REML λ̂ must not over-smooth. The bar is the same absolute
/// truth-recovery bound the mgcv quality test uses (0.18·range), R-free.
///
/// The sample size resolves both components of the surface. Preserve the
/// original recovery and complexity bars after correcting the count sampler.
#[test]
fn poisson_tensor_te_recovers_true_mean_surface_not_oversmoothed_1373() {
    init_parallelism();
    let (x, y, z, mu_true) = poisson_tensor_grid(345);
    let n = x.len();
    assert_eq!(n, 4500, "75x60 grid resolves the quadratic signal");

    let (gam_edf, prediction) = fit_te_mean(&x, &z, &y);
    let gam_mean = prediction.mean.to_vec();
    let gam_err = rmse(&gam_mean, &mu_true);
    let mut covered = 0usize;
    let mut sum_z2 = 0.0;
    for i in 0..n {
        let sd = prediction.mean_standard_error[i];
        assert!(
            sd.is_finite() && sd > 0.0,
            "finite positive mean uncertainty"
        );
        let standardized = (gam_mean[i] - mu_true[i]) / sd;
        sum_z2 += standardized * standardized;
        covered += usize::from(standardized.abs() <= 1.96);
    }
    let rms_z = (sum_z2 / n as f64).sqrt();
    let coverage_floor = (0.95 * n as f64 - 3.0 * (0.95 * 0.05 * n as f64).sqrt()).floor() as usize;
    eprintln!(
        "Poisson tensor uncertainty: covered={covered}/{n}, floor={coverage_floor}, rms_z={rms_z}"
    );
    assert!(
        covered >= coverage_floor,
        "posterior-band coverage {covered}/{n} below {coverage_floor}"
    );
    assert!(rms_z <= 2.0, "RMS standardized error {rms_z} exceeds 2");

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
         (0.18·range); gam_edf={gam_edf:.3}"
    );

    // Retain the original complexity guard: the resolved curved signal must
    // not collapse toward the {1,x}⊗{1,z} tensor penalty's null space.
    assert!(
        gam_edf >= 8.5,
        "Poisson te() effective df {gam_edf:.3} below the original 8.5 recovery bar"
    );
    assert!(
        gam_edf < 30.0,
        "Poisson te() effective df {gam_edf:.3} implausibly high (under-smoothed)"
    );
}
