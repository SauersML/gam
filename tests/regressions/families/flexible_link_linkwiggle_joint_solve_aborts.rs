//! Bug hunt (#872 / #1596): the documented `linkwiggle(...)` offset-spline
//! usage for a flexible-link binomial model. The `flexible(probit)` base link
//! fits fine on its own; adding the documented `+ linkwiggle(...)` term runs the
//! coupled exact-joint inner Newton link-warp solve.
//!
//! The learnable-link contract (#1596): a `flexible(...)` / `linkwiggle(...)`
//! request must either **engage the warp** (return a model that actually carries
//! the fitted wiggle block, with a finite in-sample deviance no worse than the
//! baseline it contains as a large-smoothing limit) OR **fail loudly** (a clear
//! `Err` the caller sees). It must NEVER silently return the no-wiggle baseline
//! — a fixed-base-link model masquerading as a flexible-link fit. The previous
//! contract here ("fits, never aborts, deviance no worse than baseline") was
//! satisfied trivially by that silent baseline fallback, which is exactly the
//! #1596 defect; this regression now forbids it.
//!
//! We deliberately use a pure-probit data-generating process (optimal offset
//! spline essentially flat — the easiest case) plus a parametric-mean case where
//! the link warp is genuinely non-redundant with the mean term.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

/// Φ(x): standard normal CDF via erf, so the data-generating link is exact probit.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

fn encode(x: &[f64], y: &[f64]) -> gam::inference::data::EncodedDataset {
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..x.len())
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn binomial_cfg() -> FitConfig {
    FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    }
}

fn in_sample_deviance(fit: &FitResult) -> f64 {
    match fit {
        FitResult::Standard(f) => f.fit.deviance,
        _ => panic!("expected a standard fit"),
    }
}

/// The documented `+ linkwiggle(...)` flexible-link model must honor the
/// learnable-link contract: it must either **engage the warp** (return a model
/// that actually carries the fitted wiggle block, with a finite deviance no
/// worse than the baseline it contains as a large-smoothing limit) OR **fail
/// loudly** (a clear `Err` the caller sees). It must NEVER silently return the
/// no-wiggle baseline masquerading as a flexible-link fit (#1596): a successful
/// fit that dropped the wiggle block is the silent contract violation this
/// regression guards against.
fn assert_wiggle_engages_or_fails_loud(
    formula_base: &str,
    ds: &gam::inference::data::EncodedDataset,
) {
    let cfg = binomial_cfg();

    let baseline = fit_from_formula(formula_base, ds, &cfg)
        .unwrap_or_else(|e| panic!("flexible(probit) baseline must fit ({formula_base}): {e:?}"));
    let dev_baseline = in_sample_deviance(&baseline);
    assert!(
        dev_baseline.is_finite(),
        "baseline deviance must be finite, got {dev_baseline}"
    );

    let wiggle_formula = format!("{formula_base} + linkwiggle(internal_knots=4)");
    match fit_from_formula(&wiggle_formula, ds, &cfg) {
        Ok(wiggle) => {
            // A *successful* flexible-link fit must carry the fitted wiggle
            // block — otherwise it is the silent baseline masquerade (#1596).
            let wiggle_engaged = match &wiggle {
                FitResult::Standard(f) => f.wiggle_knots.is_some(),
                _ => panic!("expected a standard fit"),
            };
            assert!(
                wiggle_engaged,
                "flexible(probit) + linkwiggle(...) returned a successful fit that carries \
                 NO wiggle block ({wiggle_formula}): the warp never engaged but the fit was \
                 reported as success — the silent no-wiggle baseline masquerade (#1596)."
            );
            let dev_wiggle = in_sample_deviance(&wiggle);
            assert!(
                dev_wiggle.is_finite(),
                "linkwiggle deviance must be finite, got {dev_wiggle}"
            );
            // The penalized wiggle model contains the no-wiggle baseline as a
            // large-smoothing limit, so it can never be materially worse.
            assert!(
                dev_wiggle <= dev_baseline + 1e-3 * (1.0 + dev_baseline.abs()),
                "linkwiggle deviance {dev_wiggle} must be no worse than baseline {dev_baseline}"
            );
        }
        Err(e) => {
            // Failing loud is acceptable — but it must be a *clear, actionable*
            // error about the link-wiggle non-convergence, not an obscure or
            // unrelated message, and certainly not a silent success.
            let msg = format!("{e:?}").to_lowercase();
            assert!(
                msg.contains("wiggle") || msg.contains("flexible") || msg.contains("converge"),
                "flexible(probit) + linkwiggle(...) failed ({wiggle_formula}), which is allowed, \
                 but the error must clearly name the link-wiggle non-convergence; got: {e:?}"
            );
        }
    }
}

#[test]
fn flexible_link_linkwiggle_smooth_mean_engages_or_fails_loud() {
    // Pure-probit DGP with a flexible smooth mean: the issue's exact repro.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(0);
    let ux = Uniform::new(-2.5_f64, 2.5_f64).expect("uniform");
    let uu = Uniform::new(0.0_f64, 1.0_f64).expect("uniform01");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = norm_cdf(0.8 * xi);
            if uu.sample(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();
    let ds = encode(&x, &y);
    assert_wiggle_engages_or_fails_loud("y ~ s(x) + link(type=flexible(probit))", &ds);
}

#[test]
fn flexible_link_linkwiggle_parametric_mean_engages_or_fails_loud() {
    // Parametric mean with a genuinely misspecified link: the monotone link
    // warp is NOT redundant with the (linear) mean term, so the joint solve has
    // real work to do. It must engage the warp or fail loudly — never silently
    // return the no-wiggle baseline (#1596).
    let n = 600usize;
    let mut rng = StdRng::seed_from_u64(7);
    let ux = Uniform::new(-3.0_f64, 3.0_f64).expect("uniform");
    let uu = Uniform::new(0.0_f64, 1.0_f64).expect("uniform01");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = norm_cdf(1.8 * (0.7 * xi).tanh());
            if uu.sample(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();
    let ds = encode(&x, &y);
    assert_wiggle_engages_or_fails_loud("y ~ x + link(type=flexible(probit))", &ds);
}
