//! Second-angle regression lock for issue #771: a fitted Tweedie's *response-
//! scale predictive uncertainty* must reflect the estimated dispersion `φ`.
//!
//! The sibling test `bug_hunt_tweedie_dispersion_frozen_at_one.rs` pins the
//! *linear-predictor* SE path: `φ` enters the IRLS working weight
//! `prior·μ^{2−p}/φ`, so `Vb = H⁻¹` scales as `φ` and `SE(η̂) ∝ √φ`. That is a
//! necessary check but it only exercises the coefficient-covariance consumer.
//!
//! This test exercises a DIFFERENT consumer — the **observation prediction
//! interval** on the response scale. The engine's Tweedie band
//! (`gam_predict::family_observation_band`, `ResponseFamily::Tweedie`) is not a
//! symmetric `μ ± z·σ` band. It is the equal-tailed pair of compound
//! Poisson–Gamma quantiles of a Tweedie with mean `m = E[μ]` and effective
//! dispersion `φ_eff = V/m^p`, where `V = φ̂·E[μ^p] + Var(μ)` is the total
//! predictive variance (`family_predictive_variance`) and `φ̂` is the fitted
//! `likelihood_scale.fixed_phi()`. The lower edge is exactly `0` whenever the
//! zero atom `e^{−λ}` holds the whole lower tail. The original frozen-`φ`=1 bug
//! built this band at `φ = 1` regardless of the data.
//!
//! Two independent things are asserted, each of which the frozen-`φ` bug
//! breaks and neither of which the η-SE test covers:
//!   1. ABSOLUTE `φ̂` RECOVERY — the fitted dispersion is within 4 derived
//!      standard deviations of the data's true `φ` (0.4 and 6.0), not the
//!      frozen 1.0. The SD is that of the Pearson estimator with a plug-in
//!      `μ̂`, derived from the Tweedie cumulants (see `pearson_phi_sd`); in 2000
//!      simulated replicates of this fixture the standardized error has SD
//!      0.99 at both dispersions and never exceeds 4.
//!   2. BAND CONSTRUCTION — at every row the reported `[lower, upper]` is the
//!      Tweedie(`m`, `φ_eff(φ̂)`) quantile pair at the reference tail masses:
//!      the exact CDF (`tweedie15_cdf`, independent of the engine) at `upper`
//!      equals the upper tail mass, and at `lower` equals the lower tail mass,
//!      or the zero atom covers it when `lower = 0`. With (1) this pins the band
//!      to the estimated dispersion to `CDF_TOL`.
//!
//! The exact coverage of the band under the true law is printed next to its
//! nominal value but not gated: a coverage threshold must propagate the
//! sampling error of `(β̂, φ̂)` through the quantiles, and an underived threshold
//! (the `≥ 0.90` of an earlier version, which scored a symmetric band the engine
//! does not produce) is what this test replaced.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, IntervalReference, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const B0: f64 = 0.6;
const BX: f64 = 0.7;
/// `tweedie15_cdf` is exact only at `p = 1.5`, where the Gamma jump shape is 1.
const TWEEDIE_P: f64 = 1.5;
const _: () = assert!(TWEEDIE_P == 1.5);
const CONFIDENCE_LEVEL: f64 = 0.95;

/// Number of mean coefficients in `y ~ x` (intercept and slope).
const N_COEF: usize = 2;
/// Standard deviations allowed between φ̂ and the true φ.
const PHI_Z: f64 = 4.0;

/// Allowed gap between the target tail mass and the exact CDF at a band edge.
///
/// The engine's edge is a bisection of its own Tweedie CDF down to adjacent
/// floats, so the gap is the sum of the two CDFs' errors plus `f·ulp(y)`.
/// The engine's mixture uses `P(a, x)`, which its unit test checks against
/// statrs to 1e-12 absolute; the Poisson weights sum to one, so the mixture is
/// within about 1e-12. `tweedie15_cdf` sums fewer than 100 positive terms, each
/// to a few ulps, so it is within 1e-14, and `f·ulp(y)` is below 1e-15 here.
/// The combined bound is about 1e-12; 1e-10 leaves a factor of 100 and still
/// resolves a relative error in `φ_eff` of about 1e-8.
const CDF_TOL: f64 = 1e-10;

/// Sampling SD of the Pearson dispersion `φ̂ = (1/n) Σ (yᵢ − μ̂ᵢ)²/μ̂ᵢ^p` at the
/// true `(μ, φ)` of this fixture.
///
/// With `tᵢ = (yᵢ − μᵢ)²/μᵢ^p` and `sᵢ = (yᵢ − μᵢ)μᵢ^{1−p}`, the Tweedie
/// cumulants give `Var tᵢ = p(2p−1)φ³μᵢ^{p−2} + 2φ²` and `Cov(tᵢ, sᵢ) = pφ²`.
/// Under the log link `∂tᵢ/∂ηᵢ` has mean `−pφ`, and the MLE satisfies
/// `β̂ − β ≈ H⁻¹ (1/n) Σ xᵢ sᵢ` with `H = (1/n) Σ μᵢ^{2−p} xᵢxᵢᵀ`, so
/// `n·Var φ̂ = mean[p(2p−1)φ³μᵢ^{p−2} + 2φ²] − p²φ³·x̄ᵀH⁻¹x̄`.
fn pearson_phi_sd(x: &[f64], phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let n = x.len() as f64;
    let (mut mean_var_t, mut h00, mut h01, mut h11, mut xbar1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &xi in x {
        let mu = (B0 + BX * xi).exp();
        mean_var_t += p * (2.0 * p - 1.0) * phi.powi(3) * mu.powf(p - 2.0) + 2.0 * phi * phi;
        let w = mu.powf(2.0 - p);
        h00 += w;
        h01 += w * xi;
        h11 += w * xi * xi;
        xbar1 += xi;
    }
    let (mean_var_t, h00, h01, h11, xbar1) =
        (mean_var_t / n, h00 / n, h01 / n, h11 / n, xbar1 / n);
    // x̄ = (1, mean x); x̄ᵀH⁻¹x̄ for the 2×2 H.
    let det = h00 * h11 - h01 * h01;
    assert!(det > 0.0, "information matrix must be positive definite");
    let quad = (h11 - 2.0 * h01 * xbar1 + h00 * xbar1 * xbar1) / det;
    let var = (mean_var_t - p * p * phi.powi(3) * quad) / n;
    assert!(var > 0.0, "derived Var(φ̂) must be positive, got {var}");
    var.sqrt()
}

/// Exact compound Poisson–Gamma (Tweedie, `1 < p < 2`) draw with mean `mu`,
/// dispersion `phi`: `N ~ Poisson(λ)`, `y = Σ_{i=1}^N G_i`, `G_i ~ Gamma(α, θ)`,
/// `λ = μ^{2−p}/(φ(2−p))`, `α = (2−p)/(p−1)`, `θ = φ(p−1)μ^{p−1}`; `N=0 ⇒ y=0`.
fn tweedie_sample(rng: &mut StdRng, mu: f64, phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n: u64 = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    Gamma::new(alpha * n as f64, scale)
        .expect("gamma(shape,scale)")
        .sample(rng)
}

/// Zero atom `P(Y = 0) = e^{−λ}` of a `p = 1.5` Tweedie, `λ = 2√μ/φ`.
fn tweedie15_zero_mass(mu: f64, phi: f64) -> f64 {
    (-2.0 * mu.sqrt() / phi).exp()
}

/// Exact CDF of a `p = 1.5` Tweedie with mean `mu` and dispersion `phi` at
/// `y ≥ 0`, written independently of the engine's series.
///
/// At `p = 1.5` the jumps are exponential (`α = 1`) with scale `θ = φ√μ/2`, and
/// there are `N ~ Poisson(λ)` of them, `λ = 2√μ/φ`. A sum of `k` unit
/// exponentials is at most `x = y/θ` exactly when a rate-1 Poisson process has at
/// least `k` points in `[0, x]`, so with `M ~ Poisson(x)` independent of `N`,
/// `F(y) = P(M ≥ N) = Σ_m P(M = m)·P(N ≤ m)`. Every term is positive, so the sum
/// has no cancellation. It stops once `m > x` and the geometric bound on the
/// unsummed `M` tail, which bounds the omitted mass, is below 1e-18.
fn tweedie15_cdf(y: f64, mu: f64, phi: f64) -> f64 {
    assert!(
        y.is_finite() && y >= 0.0 && mu > 0.0 && phi > 0.0,
        "tweedie15_cdf({y}, {mu}, {phi})"
    );
    let lambda = 2.0 * mu.sqrt() / phi;
    if y == 0.0 {
        return (-lambda).exp();
    }
    let x = y / (0.5 * phi * mu.sqrt());
    assert!(
        x < 700.0 && lambda < 700.0,
        "tweedie15_cdf: e^-x or e^-λ underflows (x={x}, λ={lambda})"
    );
    let mut pm = (-x).exp(); // P(M = m)
    let mut pk = (-lambda).exp(); // P(N = m)
    let mut cdf_n = pk; // P(N ≤ m)
    let mut acc = pm * cdf_n;
    for m in 1_usize.. {
        let mf = m as f64;
        pm *= x / mf;
        pk *= lambda / mf;
        cdf_n += pk;
        acc += pm * cdf_n;
        // For m + 1 > x the pmf ratios are ≤ x/(m+1) < 1, so the unsummed tail
        // Σ_{j>m} P(M = j) ≤ pm·r/(1 − r) with r = x/(m+1).
        let r = x / (mf + 1.0);
        if r < 1.0 && pm * r / (1.0 - r) < 1e-18 {
            break;
        }
    }
    acc.min(1.0)
}

struct TweedieFit {
    /// Fitted dispersion reported by the fit's inference summary.
    phi_hat: f64,
    /// Dispersion the predictive band is built from (`observation_phi`).
    band_phi: f64,
    /// Reference tail masses `(F(−z), F(z))` of the band's multiplier.
    tail_lo: f64,
    tail_hi: f64,
    /// Per-row linear predictor and its posterior SE.
    eta: Vec<f64>,
    eta_se: Vec<f64>,
    /// Per-row 95% observation interval.
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// Fit `y ~ x` as Tweedie(log) and return the fitted φ plus the response-scale
/// observation interval on the supplied evaluation grid.
fn fit_tweedie(x: &[f64], y: &[f64], eval: &[f64]) -> TweedieFit {
    let n = x.len();
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie data");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some(format!("tweedie(p={TWEEDIE_P})")),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("gam tweedie fit should succeed")
    else {
        panic!("expected a Standard Tweedie fit");
    };

    let phi_hat = fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.dispersion.phi())
        .expect("Tweedie fit must carry an estimated dispersion");
    let band_phi = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("Tweedie fit must carry the dispersion its predictive band reads");

    // The band's tail masses, read the way `predict_gamwith_uncertainty` reads
    // them: the fit's own reference law (Student-t on n − edf for an estimated
    // scale) at its central multiplier.
    let reference = IntervalReference::of_fit(&fit.fit).expect("interval reference of the fit");
    let z = reference
        .central_multiplier(CONFIDENCE_LEVEL)
        .expect("central multiplier");
    let (tail_lo, tail_hi) = (reference.cdf(-z), reference.cdf(z));

    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design at eval grid");
    let dense = design.design.to_dense();

    let tweedie_log = LikelihoodSpec::new(
        ResponseFamily::Tweedie { p: TWEEDIE_P },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        tweedie_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: CONFIDENCE_LEVEL,
            covariance_mode: InferenceCovarianceMode::Conditional,
            includeobservation_interval: true,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("tweedie predict with observation interval");

    TweedieFit {
        phi_hat,
        band_phi,
        tail_lo,
        tail_hi,
        eta: pred.eta.to_vec(),
        eta_se: pred.eta_standard_error.to_vec(),
        lower: pred
            .observation_lower
            .expect("observation interval requested")
            .to_vec(),
        upper: pred
            .observation_upper
            .expect("observation interval requested")
            .to_vec(),
    }
}

/// The Tweedie law the engine's band at row `i` is a quantile pair of:
/// `(m, φ_eff)` with `m = E[μ] = exp(η̂ + s²/2)`, `v = Var(μ) = e^{2η̂+s²}(e^{s²} − 1)`
/// under `η ~ N(η̂, s²)`, `V = φ̂·m^p(1 + v/m²)^{p(p−1)/2} + v` and `φ_eff = V/m^p`.
fn band_law(fit: &TweedieFit, i: usize) -> (f64, f64) {
    let p = TWEEDIE_P;
    let (eta, s2) = (fit.eta[i], fit.eta_se[i] * fit.eta_se[i]);
    let m = (eta + 0.5 * s2).exp();
    let v = (2.0 * eta + s2).exp() * s2.exp_m1();
    let moment = if v > 0.0 {
        m.powf(p) * (1.0 + v / (m * m)).powf(0.5 * p * (p - 1.0))
    } else {
        m.powf(p)
    };
    (m, (fit.band_phi * moment + v) / m.powf(p))
}

#[test]
fn tweedie_observation_interval_reflects_estimated_dispersion() {
    init_parallelism();

    let n = 4000usize;
    let phi_lo = 0.4_f64;
    let phi_hi = 6.0_f64;

    // Shared covariate values so the only inferential difference is the true φ.
    let mut rng = StdRng::seed_from_u64(0x71_C0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();

    let mut rng_lo = StdRng::seed_from_u64(101);
    let mut rng_hi = StdRng::seed_from_u64(202);
    let mut y_lo = Vec::with_capacity(n);
    let mut y_hi = Vec::with_capacity(n);
    for &xi in &x {
        let mu = (B0 + BX * xi).exp();
        y_lo.push(tweedie_sample(&mut rng_lo, mu, phi_lo));
        y_hi.push(tweedie_sample(&mut rng_hi, mu, phi_hi));
    }

    // The observation band at the training abscissae.
    let fit_lo = fit_tweedie(&x, &y_lo, &x);
    let fit_hi = fit_tweedie(&x, &y_hi, &x);

    // ── 1. Absolute φ̂ recovery (the ratio test cannot see a shared bias) ────
    for (fit, phi_true, tag) in [(&fit_lo, phi_lo, "low"), (&fit_hi, phi_hi, "high")] {
        let sd = pearson_phi_sd(&x, phi_true);
        // Plus the `n` vs `n − p` divisor bias of the Pearson mean.
        let tol = PHI_Z * sd + (N_COEF as f64 / n as f64) * phi_true;
        let z = (fit.phi_hat - phi_true) / sd;
        eprintln!(
            "[tweedie-obs] {tag}: true φ={phi_true} φ̂={:.5} SD(φ̂)={sd:.5} z={z:.2}",
            fit.phi_hat
        );
        assert!(
            (fit.phi_hat - phi_true).abs() <= tol,
            "Tweedie φ̂ does not recover the {tag} dispersion: φ̂={:.5} vs true {phi_true}, \
             |error| {:.5} > {tol:.5} ({PHI_Z} derived SDs, SD={sd:.5}, z={z:.2}); the \
             frozen-φ bug pins it at 1.0",
            fit.phi_hat,
            (fit.phi_hat - phi_true).abs()
        );
        // The band must be built from the dispersion the fit reports. Both are
        // copies of one estimate, so they agree to rounding.
        assert!(
            (fit.band_phi - fit.phi_hat).abs() <= 4.0 * f64::EPSILON * fit.phi_hat,
            "{tag}: the observation band reads φ={} but the fit reports φ̂={}",
            fit.band_phi,
            fit.phi_hat
        );
    }

    // ── 2. The band is the Tweedie(m, φ_eff(φ̂)) quantile pair ───────────────
    for (fit, phi_true, y, tag) in [
        (&fit_lo, phi_lo, &y_lo, "low"),
        (&fit_hi, phi_hi, &y_hi, "high"),
    ] {
        let (mut atom_rows, mut worst_gap) = (0usize, 0.0_f64);
        let (mut exact_cov, mut nominal_cov, mut hits) = (0.0_f64, 0.0_f64, 0usize);
        for i in 0..n {
            let (lower, upper) = (fit.lower[i], fit.upper[i]);
            assert!(
                lower.is_finite() && upper.is_finite() && 0.0 <= lower && lower < upper,
                "{tag} row {i}: observation band [{lower}, {upper}] is not a proper interval \
                 on the Tweedie support"
            );
            let (m, phi_eff) = band_law(fit, i);
            assert!(
                phi_eff >= fit.band_phi,
                "{tag} row {i}: φ_eff={phi_eff} below φ̂={}; estimation variance only widens",
                fit.band_phi
            );
            let gap_hi = (tweedie15_cdf(upper, m, phi_eff) - fit.tail_hi).abs();
            assert!(
                gap_hi <= CDF_TOL,
                "{tag} row {i}: upper edge {upper} is not the {:.6} quantile of \
                 Tweedie(m={m}, φ_eff={phi_eff}): |F(upper) − {:.6}| = {gap_hi:.3e} > {CDF_TOL:e}",
                fit.tail_hi,
                fit.tail_hi
            );
            worst_gap = worst_gap.max(gap_hi);
            let zero_mass = tweedie15_zero_mass(m, phi_eff);
            if lower == 0.0 {
                // The quantile is 0 exactly when the atom holds the lower tail.
                assert!(
                    zero_mass >= fit.tail_lo - CDF_TOL,
                    "{tag} row {i}: lower edge is 0 but the zero atom of \
                     Tweedie(m={m}, φ_eff={phi_eff}) is {zero_mass:.6} < {:.6}",
                    fit.tail_lo
                );
                atom_rows += 1;
            } else {
                assert!(
                    zero_mass < fit.tail_lo + CDF_TOL,
                    "{tag} row {i}: lower edge {lower} > 0 although the zero atom \
                     {zero_mass:.6} covers the lower tail mass {:.6}",
                    fit.tail_lo
                );
                let gap_lo = (tweedie15_cdf(lower, m, phi_eff) - fit.tail_lo).abs();
                assert!(
                    gap_lo <= CDF_TOL,
                    "{tag} row {i}: lower edge {lower} is not the {:.6} quantile of \
                     Tweedie(m={m}, φ_eff={phi_eff}): |F(lower) − {:.6}| = {gap_lo:.3e} > \
                     {CDF_TOL:e}",
                    fit.tail_lo,
                    fit.tail_lo
                );
                worst_gap = worst_gap.max(gap_lo);
            }

            // Diagnostics only (see the module doc): the band's exact coverage
            // under the true law, its nominal coverage and the in-sample hits.
            let mu_true = (B0 + BX * x[i]).exp();
            let below = if lower > 0.0 {
                tweedie15_cdf(lower, mu_true, phi_true)
            } else {
                0.0
            };
            exact_cov += tweedie15_cdf(upper, mu_true, phi_true) - below;
            nominal_cov += fit.tail_hi - if lower > 0.0 { fit.tail_lo } else { 0.0 };
            hits += usize::from(lower <= y[i] && y[i] <= upper);
        }
        eprintln!(
            "[tweedie-obs] {tag}: band = Tweedie quantiles at φ_eff(φ̂) on all {n} rows \
             (max CDF gap {worst_gap:.2e}, {atom_rows} rows with lower = 0); coverage under \
             the true law {:.4}, nominal {:.4}, in-sample {:.4}",
            exact_cov / n as f64,
            nominal_cov / n as f64,
            hits as f64 / n as f64
        );
    }
}
