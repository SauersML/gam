//! Regression tests for #2026: a Tweedie fit must never silently use a variance
//! power the caller did not choose.
//!
//! BUG (#2026): `resolve_family` mapped a bare `family="tweedie"` to
//! `ResponseFamily::Tweedie { p: 1.5 }` and the fit used that fixed power
//! unconditionally. On data whose true power `p ≠ 1.5` the fitted mean is robust
//! (log-link quasi-likelihood), but the conditional variance `Var(Y|x) = φ μ^p`
//! — and every observation interval derived from it — is miscalibrated because
//! `p` is wrong, and nothing told the caller.
//!
//! HISTORY: #2026 first answered this by profiling `p` over `(1, 2)` for a bare
//! `tweedie`, and these tests asserted that the profiled power tracked the truth.
//! a893d85bc retired that profile because it is a derivative-free hyperparameter
//! search, which SPEC.md forbids. The shipped contract is now: a bare
//! `tweedie`/`tw` is REFUSED with a typed configuration error naming the explicit
//! form, and an explicit `tweedie(p)` pins `p` exactly. The #2026 defect (a
//! silent default power) therefore cannot recur. These tests pin that contract,
//! on the same compound-Poisson-gamma DGPs the estimator tests used.
//!
//! DGP: Tweedie compound-Poisson-gamma (Jørgensen) with a correctly-specified
//! log-linear mean `log μ = 0.5 + 0.8·x`, x ~ U(0,1), `φ = 1`.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::types::ResponseFamily;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 800;
const P_TRUE: f64 = 1.8;
const PHI: f64 = 1.0;
const SEED: u64 = 2_026_018;

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

/// Simulate a compound-Poisson-gamma (Jørgensen) Tweedie sample at `p_true`
/// with mean `true_mu(x)` and dispersion `PHI`.
fn simulate_tweedie(p_true: f64, n: usize, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif01 = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi: f64 = unif01.sample(&mut rng);
        let mu = true_mu(xi);
        let lambda = mu.powf(2.0 - p_true) / (PHI * (2.0 - p_true));
        let shape = (2.0 - p_true) / (p_true - 1.0);
        let scale = PHI * (p_true - 1.0) * mu.powf(p_true - 1.0);
        let n_jumps = poisson_sample(lambda, &mut rng, &unif01);
        let mut yi = 0.0;
        for _ in 0..n_jumps {
            yi += gamma_sample(shape, scale, &mut rng);
        }
        x.push(xi);
        y.push(yi);
    }
    encode(&[("x", &x), ("y", &y)])
}

/// Fit `y ~ x` with the given family string and return the power the fitted
/// engine family carries.
fn fitted_tweedie_power(family: &str, ds: &EncodedDataset) -> f64 {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ x", ds, &cfg)
        .unwrap_or_else(|e| panic!("explicit {family} fit failed: {e:?}"))
    else {
        panic!("Tweedie(log) is a scalar GLM => expected FitResult::Standard for {family}");
    };
    recovered_power(&fit)
}

#[test]
fn bare_tweedie_is_refused_instead_of_silently_fitting_a_default_power() {
    init_parallelism();

    let ds = simulate_tweedie(P_TRUE, N, SEED);

    // ---- bare tweedie: refused, never fitted at a power nobody chose --------
    for bare in ["tweedie", "tw"] {
        let cfg = FitConfig {
            family: Some(bare.to_string()),
            ..FitConfig::default()
        };
        let err = match fit_from_formula("y ~ x", &ds, &cfg) {
            Ok(FitResult::Standard(fit)) => panic!(
                "bare family=\"{bare}\" fitted at p={} instead of refusing; a893d85bc \
                 requires an explicit power, and a silent default is the #2026 defect",
                recovered_power(&fit)
            ),
            Ok(_) => panic!("bare family=\"{bare}\" fitted instead of refusing"),
            Err(err) => format!("{err:?}"),
        };
        assert!(
            err.contains("explicit variance power"),
            "bare family=\"{bare}\" must be refused with the typed explicit-power \
             configuration error, got: {err}"
        );
    }

    // ---- explicit tweedie(p): the pinned power is preserved verbatim ---------
    let p_fixed = fitted_tweedie_power("tweedie(1.4)", &ds);
    eprintln!("tweedie #2026 (seed {SEED}, p_true={P_TRUE}): n={N} fixed_p={p_fixed:.4}");
    assert!(
        (p_fixed - 1.4).abs() < 1e-12,
        "explicit tweedie(1.4) must pin p exactly; got {p_fixed}"
    );
    let p_named = fitted_tweedie_power(&format!("tweedie(p={P_TRUE})"), &ds);
    assert!(
        (p_named - P_TRUE).abs() < 1e-12,
        "explicit tweedie(p={P_TRUE}) must pin p exactly; got {p_named}"
    );
}

/// #2064 removed the fixed power grid {1.1, …, 1.9}. The power is a continuous
/// parameter: a truth that sits OFF the old grid nodes (`p = 1.65`, halfway
/// between 1.6 and 1.7) must be carried through the fit verbatim when requested
/// explicitly, not snapped to a node.
#[test]
fn explicit_off_grid_tweedie_power_is_pinned_verbatim() {
    init_parallelism();

    const P_OFF_GRID: f64 = 1.65;
    let ds = simulate_tweedie(P_OFF_GRID, 3000, 2_064_017);
    let p_hat = fitted_tweedie_power(&format!("tweedie(p={P_OFF_GRID})"), &ds);
    eprintln!(
        "tweedie #2064: p_true={P_OFF_GRID} n=3000 fitted_p={p_hat:.6}; \
         the removed grid had no node here (nearest 1.6 / 1.7)"
    );
    assert!(
        (p_hat - P_OFF_GRID).abs() < 1e-12,
        "explicit off-grid tweedie(p={P_OFF_GRID}) was not pinned verbatim: fitted p={p_hat}"
    );
}
