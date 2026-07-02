//! Regression test for #2026: a bare `family="tweedie"` must ESTIMATE the
//! variance power `p` from the data (mgcv `tw()` semantics), not silently fit
//! at the hardcoded fallback `p = 1.5`.
//!
//! BUG (before this fix): `resolve_family` mapped a bare `family="tweedie"` to
//! `ResponseFamily::Tweedie { p: 1.5 }` and the fit used that fixed power
//! unconditionally. On data whose true power `p ≠ 1.5` the fitted mean is robust
//! (log-link quasi-likelihood), but the conditional variance `Var(Y|x) = φ μ^p`
//! — and every observation interval derived from it — is miscalibrated because
//! `p` is wrong. mgcv's `tw()` profiles `p` over `(1, 2)`; gam did not.
//!
//! FIX (#2026): a bare `family="tweedie"`/`"tw"` (no explicit power) now profiles
//! `p` by maximum saddlepoint likelihood over `(1, 2)` before the reported fit,
//! so the recovered power tracks the data. An explicit `tweedie(1.6)` still pins
//! `p`.
//!
//! DGP: Tweedie compound-Poisson-gamma (Jørgensen) with a correctly-specified
//! log-linear mean `log μ = 0.5 + 0.8·x`, x ~ U(0,1), true power `p_true = 1.8`,
//! `φ = 1`. METRIC: the recovered power `p̂` reported on the fitted family must
//! land within `TOL` of `p_true`. This FAILS before the fix (`p̂ ≡ 1.5`, error
//! `|1.5 − 1.8| = 0.3 > TOL`) and PASSES after (profile recovery).

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam::types::ResponseFamily;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 800;
const P_TRUE: f64 = 1.8;
const PHI: f64 = 1.0;
const SEED: u64 = 2_026_018;
/// Recovery tolerance. The old hardcoded fallback `p = 1.5` misses by
/// `|1.5 − 1.8| = 0.3`, so any tolerance `< 0.3` fails before the fix; the
/// profile estimator recovers `p_true` comfortably inside `0.2` after it.
const TOL: f64 = 0.2;

fn true_mu(x: f64) -> f64 {
    (0.5 + 0.8 * x).exp()
}

fn encode(cols: &[(&str, &[f64])]) -> EncodedDataset {
    let n = cols[0].1.len();
    let headers: Vec<String> = cols.iter().map(|(h, _)| (*h).to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(
                cols.iter()
                    .map(|(_, c)| c[i].to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie dataset")
}

/// Knuth Poisson sampler — adequate for the moderate-λ Tweedie DGP.
fn poisson_sample(lambda: f64, rng: &mut StdRng, unif: &Uniform<f64>) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0;
    loop {
        p *= unif.sample(rng);
        if p <= l {
            return k;
        }
        k += 1;
        if k > 10_000 {
            return k;
        }
    }
}

/// Marsaglia–Tsang gamma sampler (shape > 0) with the given scale.
fn gamma_sample(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    let normal = Normal::new(0.0, 1.0).expect("normal");
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    if shape < 1.0 {
        let u: f64 = unif.sample(rng);
        return gamma_sample(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z: f64 = normal.sample(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = unif.sample(rng);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * scale;
        }
    }
}

/// Recover the Tweedie variance power carried on a fitted family.
fn recovered_power(fit: &gam::StandardFitResult) -> f64 {
    match fit
        .fit
        .likelihood_family
        .as_ref()
        .expect("standard Tweedie fit reports an engine family")
        .response
    {
        ResponseFamily::Tweedie { p } => p,
        ref other => panic!("expected a Tweedie response family, got {other:?}"),
    }
}

#[test]
fn bare_tweedie_estimates_variance_power_from_data() {
    init_parallelism();

    // ---- Tweedie compound-Poisson-gamma DGP (Jørgensen), true p = 1.8 --------
    let mut rng = StdRng::seed_from_u64(SEED);
    let unif01 = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi: f64 = unif01.sample(&mut rng);
        let mu = true_mu(xi);
        let lambda = mu.powf(2.0 - P_TRUE) / (PHI * (2.0 - P_TRUE));
        let shape = (2.0 - P_TRUE) / (P_TRUE - 1.0);
        let scale = PHI * (P_TRUE - 1.0) * mu.powf(P_TRUE - 1.0);
        let n_jumps = poisson_sample(lambda, &mut rng, &unif01);
        let mut yi = 0.0;
        for _ in 0..n_jumps {
            yi += gamma_sample(shape, scale, &mut rng);
        }
        x.push(xi);
        y.push(yi);
    }

    let ds = encode(&[("x", &x), ("y", &y)]);

    // ---- bare tweedie: p must be ESTIMATED (mgcv tw() semantics) -------------
    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &ds, &cfg).expect("bare tweedie fit");
    let FitResult::Standard(fit) = result else {
        panic!("Tweedie(log) is a scalar GLM => expected FitResult::Standard");
    };
    let p_hat = recovered_power(&fit);

    // ---- explicit tweedie(p): the pinned power is preserved verbatim ---------
    let cfg_fixed = FitConfig {
        family: Some("tweedie(1.4)".to_string()),
        ..FitConfig::default()
    };
    let result_fixed = fit_from_formula("y ~ x", &ds, &cfg_fixed).expect("pinned tweedie fit");
    let FitResult::Standard(fit_fixed) = result_fixed else {
        panic!("expected FitResult::Standard for tweedie(1.4)");
    };
    let p_fixed = recovered_power(&fit_fixed);

    eprintln!(
        "tweedie #2026 (seed {SEED}, p_true={P_TRUE}): n={N} \
         bare_p_hat={p_hat:.4} (err {:.4}) fixed_p={p_fixed:.4} \
         old_fallback_err={:.4}",
        (p_hat - P_TRUE).abs(),
        (1.5 - P_TRUE).abs(),
    );

    // An explicit power is never touched by the estimator.
    assert!(
        (p_fixed - 1.4).abs() < 1e-9,
        "explicit tweedie(1.4) must pin p exactly; got {p_fixed}"
    );

    // The bare-tweedie power must be ESTIMATED toward the truth, not left at the
    // hardcoded 1.5 fallback. Before #2026 this is exactly 1.5 (err 0.3 > TOL).
    assert!(
        (p_hat - P_TRUE).abs() < TOL,
        "bare family=\"tweedie\" did not estimate the variance power: recovered p̂={p_hat:.4} \
         is {:.4} from the true p={P_TRUE} (tolerance {TOL}). The pre-#2026 hardcoded \
         fallback p=1.5 gives error {:.4}; a working profile estimator must beat it.",
        (p_hat - P_TRUE).abs(),
        (1.5 - P_TRUE).abs(),
    );
}
