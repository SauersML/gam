//! Bug hunt (#1596 / #1598): the learnable / flexible / blended binomial link
//! cluster.
//!
//! #1596 — `link(type=flexible(logit))` on a parametric, identifiable binomial
//! predictor silently fell back to plain logit: the coupled link-wiggle joint
//! Newton solve failed KKT certification, and a suppressed `log::warn!` fallback
//! returned the no-wiggle baseline `Ok(...)`, bit-identical to a fixed-logit fit
//! (no wiggle block, deviance unchanged). A flexible-link request must NEVER
//! silently return the fixed base link: it must either engage the warp (and, on
//! link-misspecified data, strictly improve) or fail loudly.
//!
//! #1598 — `link(type=blended(...))` / `mixture(...)`: the formula / Python path
//! never threaded the parsed mixture spec into `FitOptions.mixture_link`, so the
//! solver guard aborted with "BinomialMixture requires mixture_link
//! specification" before fitting. The spec must thread through so the request
//! reaches the solver (then either fits or fails with a clear solver error — not
//! a pre-solve wiring abort).

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

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

fn standard_deviance(fit: &FitResult) -> f64 {
    match fit {
        FitResult::Standard(f) => f.fit.deviance,
        _ => panic!("expected a standard fit"),
    }
}

fn wiggle_engaged(fit: &FitResult) -> bool {
    match fit {
        FitResult::Standard(f) => f.wiggle_knots.is_some(),
        _ => false,
    }
}

/// #1596: cloglog-misspecified, parametric (identifiable) predictor. A
/// `flexible(logit)` request must not silently masquerade as plain logit.
#[test]
fn flexible_logit_binomial_parametric_not_silent_logit_noop_1596() {
    let n = 2000usize;
    let mut rng = StdRng::seed_from_u64(3);
    let ux = Uniform::new(-2.5_f64, 2.5_f64).expect("uniform");
    let uu = Uniform::new(0.0_f64, 1.0_f64).expect("uniform01");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    // TRUE link = cloglog; logit is grossly misspecified, so a working warp must
    // help. p = 1 - exp(-exp(eta)).
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = -0.5 + 1.8 * xi;
            let p = (1.0 - (-(eta.exp())).exp()).clamp(1e-4, 1.0 - 1e-4);
            if uu.sample(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();
    let ds = encode(&x, &y);
    let cfg = binomial_cfg();

    // Plain logit baseline (the masquerade the flexible fit must not silently be).
    let plain = fit_from_formula("y ~ x", &ds, &cfg).expect("plain logit must fit");
    let dev_plain = standard_deviance(&plain);
    assert!(dev_plain.is_finite());

    match fit_from_formula("y ~ x + link(type=flexible(logit))", &ds, &cfg) {
        Ok(flex) => {
            // A *successful* flexible fit must (a) carry the wiggle block — not
            // be the silent no-wiggle baseline — and (b) on this grossly
            // link-misspecified data actually help (strictly lower deviance).
            // A negligible improvement is the reward-hack we forbid.
            assert!(
                wiggle_engaged(&flex),
                "flexible(logit) returned a successful fit with NO wiggle block: the silent \
                 no-wiggle logit masquerade (#1596)."
            );
            let dev_flex = standard_deviance(&flex);
            assert!(dev_flex.is_finite());
            assert!(
                dev_flex < dev_plain - 1.0,
                "flexible(logit) engaged but did not materially improve over plain logit on \
                 cloglog-misspecified data: dev_flex={dev_flex} vs dev_plain={dev_plain}. The \
                 warp must genuinely engage, not perturb negligibly (#1596)."
            );
        }
        Err(e) => {
            // Failing loud is acceptable; a silent baseline is not. The error
            // must clearly name the link-wiggle non-convergence.
            let msg = format!("{e:?}").to_lowercase();
            assert!(
                msg.contains("wiggle") || msg.contains("flexible") || msg.contains("converge"),
                "flexible(logit) failed (allowed), but the error must clearly name the \
                 link-wiggle non-convergence; got: {e:?}"
            );
        }
    }
}

/// #1598: the blended/mixture spec must thread into the solver. The fit may then
/// succeed OR fail with a clear *solver* error — but it must not abort *before*
/// the solver with the wiring guard "requires mixture_link specification".
#[test]
fn blended_mixture_link_threads_spec_not_wiring_abort_1598() {
    let n = 1500usize;
    let mut rng = StdRng::seed_from_u64(7);
    let ux = Uniform::new(-2.0_f64, 2.0_f64).expect("uniform");
    let uu = Uniform::new(0.0_f64, 1.0_f64).expect("uniform01");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    // Clean logit truth; blended(logit, probit) nests logit so it should fit.
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = 1.0 / (1.0 + (-(0.3 + 1.5 * xi)).exp());
            if uu.sample(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();
    let ds = encode(&x, &y);
    let cfg = binomial_cfg();

    match fit_from_formula("y ~ x + link(type=blended(logit, probit))", &ds, &cfg) {
        Ok(fit) => {
            assert!(standard_deviance(&fit).is_finite());
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                !msg.contains("requires mixture_link specification"),
                "blended/mixture link spec was NOT threaded into FitOptions — the request \
                 aborted at the pre-solve wiring guard instead of reaching the solver (#1598); \
                 got: {e:?}"
            );
            // Probit's CDF can saturate; just confirm we reached a real solver
            // diagnostic (anything other than the wiring guard is acceptable as
            // a clear, actionable error per the issue's contract).
            let _ = msg;
        }
    }
}
