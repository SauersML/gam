//! GitHub issue #1065 — magic family auto-detection for count responses.
//!
//! When the user omits `family=` (the `"auto"` default), gam infers the
//! likelihood from the response column. Before this change, a non-negative
//! integer COUNT response (`0,1,2,3,7,12,...`) that was not strictly `{0,1}`
//! silently fell to Gaussian + identity, when the obvious GLM default —
//! the one mgcv/statsmodels users expect — is Poisson + log.
//!
//! This test pins the magic-by-default contract at two layers:
//!
//!   1. The single inference seam `resolve_family` (every entry point routes
//!      through it). With NO family supplied it must pick:
//!        * count `{0,1,2,...}` with a value ≥ 2  → Poisson, log link
//!        * genuinely continuous (fractional) data → Gaussian, identity link
//!        * strictly binary `{0,1}`               → Binomial, logit link
//!        * a negative integer mixed in           → Gaussian (conservative)
//!
//!   2. End-to-end OBJECTIVE quality: an `family="auto"` fit of
//!      `y ~ s(x)` on Poisson-generated count data must RECOVER the known
//!      log-intensity `eta = log E[y|x]`. A Gaussian-identity fit (the old
//!      misfire) reconstructs the mean on the natural scale, not `exp(eta)`,
//!      so recovering the centered `eta` shape in eta-units is direct evidence
//!      that auto-detection actually selected Poisson(log). We assert the
//!      auto fit's centered-eta recovery RMSE is a small fraction of the
//!      signal range AND strictly better than a deliberately-wrong
//!      Gaussian-identity fit scored on the same eta-recovery metric.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, ResponseColumnKind, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    resolve_family,
};
use ndarray::{Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};

const N: usize = 300;
const SEED: u64 = 1065;

fn truth_eta(x: f64) -> f64 {
    // log-intensity with a clear non-linear shape; exp(eta) stays in a sane
    // count range over x in [0, 10].
    let pi = std::f64::consts::PI;
    1.0 + 0.7 * (x * pi / 5.0).sin()
}

fn centered(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    v.iter().map(|x| x - mean).collect()
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(1) as f64;
    (a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / n).sqrt()
}

/// Layer 1: the single auto-inference seam. With no family supplied,
/// `resolve_family` must read the response signature and pick the principled
/// GLM default — without misfiring on continuous or binary data.
#[test]
fn resolve_family_auto_count_picks_poisson_log() {
    // Non-negative integer counts reaching beyond {0,1} → Poisson(log).
    let counts = array![0.0, 1.0, 2.0, 3.0, 0.0, 5.0, 1.0, 12.0, 4.0, 2.0];
    let resolved = resolve_family(
        None,
        None,
        None,
        counts.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("auto inference on a count response should succeed");
    assert_eq!(
        resolved.response,
        ResponseFamily::Poisson,
        "non-negative integer counts with a value >= 2 must auto-infer Poisson"
    );
    assert_eq!(
        resolved.link,
        InverseLink::Standard(StandardLink::Log),
        "auto-inferred Poisson must use the canonical log link"
    );

    // Integer counts arriving as f64 round-trips (CSV-style) must still detect.
    let noisy_counts = array![0.0, 1.0 + 1e-12, 2.0 - 5e-10, 7.0, 3.0 + 1e-10];
    let resolved = resolve_family(
        None,
        None,
        None,
        noisy_counts.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("auto inference on near-integer counts should succeed");
    assert_eq!(
        resolved.response,
        ResponseFamily::Poisson,
        "near-integer counts within the round tolerance must auto-infer Poisson"
    );
}

#[test]
fn resolve_family_auto_continuous_stays_gaussian() {
    // Genuinely continuous (fractional) data must NOT misfire to Poisson.
    let continuous = array![0.3, 1.7, 2.25, 3.9, 0.01, 5.5, 1.42, 12.8];
    let resolved = resolve_family(
        None,
        None,
        None,
        continuous.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("auto inference on continuous data should succeed");
    assert_eq!(
        resolved.response,
        ResponseFamily::Gaussian,
        "fractional continuous data must stay Gaussian, not misfire to Poisson"
    );
    assert_eq!(
        resolved.link,
        InverseLink::Standard(StandardLink::Identity),
        "auto Gaussian must keep the identity link"
    );

    // A single negative integer disqualifies the count signature: conservative
    // Gaussian fallback (Poisson support is non-negative).
    let signed = array![0.0, 1.0, 2.0, -1.0, 3.0];
    let resolved = resolve_family(
        None,
        None,
        None,
        signed.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("auto inference on signed integers should succeed");
    assert_eq!(
        resolved.response,
        ResponseFamily::Gaussian,
        "a negative value must fall back to Gaussian, never Poisson"
    );
}

#[test]
fn resolve_family_auto_binary_stays_binomial() {
    // Strictly {0,1} must remain Binomial(logit) — the count rule requires a
    // value >= 2, so it never steals the binary case.
    let binary = array![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let resolved = resolve_family(
        None,
        None,
        None,
        binary.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("auto inference on binary data should succeed");
    assert_eq!(
        resolved.response,
        ResponseFamily::Binomial,
        "strictly {{0,1}} data must stay Binomial, not be captured by the count rule"
    );
    assert_eq!(
        resolved.link,
        InverseLink::Standard(StandardLink::Logit),
        "auto Binomial must keep the logit link"
    );
}

/// Layer 2: end-to-end. `family="auto"` on Poisson count data must select
/// Poisson(log) and recover the generating log-intensity — strictly better
/// than a forced Gaussian-identity fit on the same counts.
#[test]
fn auto_fit_on_counts_recovers_log_intensity() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform 0..10");
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_truth = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let eta = truth_eta(a);
        let pois = Poisson::new(eta.exp()).expect("poisson mean > 0");
        let count: f64 = pois.sample(&mut rng);
        x.push(a);
        eta_truth.push(eta);
        y.push(count);
    }

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode count dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // --- AUTO fit: family omitted (the "auto" default) -------------------
    let auto_cfg = FitConfig::default();
    assert!(
        auto_cfg.family.is_none(),
        "the default FitConfig must leave family unset so auto-detection runs"
    );
    let auto = fit_from_formula("y ~ s(x, k=8)", &ds, &auto_cfg).expect("auto count fit");
    let FitResult::Standard(auto_fit) = auto else {
        panic!("auto fit on count data must select a standard GLM family (Poisson)");
    };

    // Rebuild the design and read eta_hat = design*beta (log link → eta is the
    // linear predictor). A Gaussian-identity misfire would instead place the
    // MEAN on this scale, so recovering eta directly is the discriminating test.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
    }
    let design = build_term_collection_design(grid.view(), &auto_fit.resolvedspec)
        .expect("rebuild auto design");
    let auto_eta: Vec<f64> = design.design.apply(&auto_fit.fit.beta).to_vec();

    // --- WRONG Gaussian-identity fit on the same counts (the old misfire) -
    let gauss_cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let gauss = fit_from_formula("y ~ s(x, k=8)", &ds, &gauss_cfg).expect("gaussian count fit");
    let FitResult::Standard(gauss_fit) = gauss else {
        panic!("forced gaussian fit must be a standard GLM");
    };
    let gauss_design = build_term_collection_design(grid.view(), &gauss_fit.resolvedspec)
        .expect("rebuild gaussian design");
    // Gaussian-identity: design*beta IS the fitted mean; the comparable
    // log-intensity estimate is log(mu). Guard the log against the identity
    // model's unconstrained (possibly non-positive) fitted mean.
    let gauss_mu: Vec<f64> = gauss_design.design.apply(&gauss_fit.fit.beta).to_vec();
    let gauss_eta: Vec<f64> = gauss_mu.iter().map(|m| m.max(1e-6).ln()).collect();

    // Score both on centered eta-recovery (eta is identifiable up to an
    // additive constant; the SHAPE lives in the centered vector).
    let ct = centered(&eta_truth);
    let auto_rmse = rmse(&centered(&auto_eta), &ct);
    let gauss_rmse = rmse(&centered(&gauss_eta), &ct);

    let signal_range = {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &e in &ct {
            lo = lo.min(e);
            hi = hi.max(e);
        }
        hi - lo
    };

    println!(
        "auto-Poisson eta-RMSE = {auto_rmse:.4}, gaussian-identity eta-RMSE = {gauss_rmse:.4}, signal range = {signal_range:.4}"
    );

    // (a) the auto fit recovers the log-intensity shape to a small fraction of
    //     its range — only achievable if Poisson(log) was actually selected.
    assert!(
        auto_rmse < 0.20 * signal_range,
        "auto fit should recover the log-intensity (RMSE {auto_rmse:.4} vs range {signal_range:.4}); \
         a Gaussian-identity misfire would not"
    );
    // (b) and it strictly beats the wrong Gaussian-identity model on the
    //     log-intensity recovery metric.
    assert!(
        auto_rmse < gauss_rmse,
        "auto (Poisson) eta-recovery {auto_rmse:.4} must beat gaussian-identity {gauss_rmse:.4}"
    );
}
