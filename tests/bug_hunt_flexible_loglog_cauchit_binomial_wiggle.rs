//! #2155 regression (Rust / library angle): the binomial joint link-wiggle
//! ("flexible link") solver must fit `flexible(loglog)` and `flexible(cauchit)`,
//! not accept-then-crash. The committed Python test pins the graceful-handling
//! contract through the gamfit API (fit-and-predict OR clean rejection, never an
//! opaque IntegrationError); this one drives the FULL `fit_from_formula`
//! pipeline directly and pins the *stronger* outcome the fix actually delivers:
//! the warp genuinely engages and converges for both links.
//!
//! Root cause: two wiggle-support gate predicates (`inverse_link_supports_joint_
//! wiggle`, `binomial_inverse_link_supports_joint_wiggle`) listed only
//! logit/probit/cloglog while the permissive parse gate admitted loglog/cauchit,
//! so the config was accepted then aborted deep in the solver. The wiggle kernel
//! evaluates the base inverse link through generic jets that already carry
//! LogLog/Cauchit, so teaching the gates makes the fit real.
//!
//! Asserted, from an angle distinct from the Python predict check:
//!   * cloglog control — flexible fit converges with the warp engaged,
//!   * loglog / cauchit — each fit converges (`outer_converged`), the monotone
//!     warp is actually engaged (`wiggle_degree`/`wiggle_knots` present, not a
//!     silent drop to the fixed base link), and the deviance is finite,
//!   * the loglog and cauchit warped fits are genuinely distinct from cloglog
//!     (distinct fitted coefficients) — a gate re-restriction or a silent
//!     fallback to a fixed base link would collapse these.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

// deterministic splitmix64
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn binomial_data(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut r = Rng(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * r.u();
        let p = 1.0 / (1.0 + (-(0.8 * x)).exp());
        let y = if r.u() < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![y.to_string(), x.to_string()]));
    }
    encode_recordswith_inferred_schema(["y", "x"].iter().map(|s| s.to_string()).collect(), rows)
        .expect("encode binomial dataset")
}

/// Fit `y ~ x + link(type=flexible(<link>))` as a binomial GAM and return
/// `(outer_converged, wiggle_degree, wiggle_knots_len, deviance, beta)`.
fn fit_flexible(link: &str) -> (bool, Option<usize>, Option<usize>, f64, Vec<f64>) {
    let d = binomial_data(600, 2155);
    let cfg = FitConfig {
        family: Some("binomial".into()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ x + link(type=flexible({link}))");
    let FitResult::Standard(fit) = fit_from_formula(&formula, &d, &cfg)
        .unwrap_or_else(|e| panic!("binomial flexible({link}) must fit (#2155), got: {e}"))
    else {
        panic!("expected a standard binomial GAM fit for flexible({link})");
    };
    (
        fit.fit.outer_converged,
        fit.wiggle_degree,
        fit.wiggle_knots.as_ref().map(|k| k.len()),
        fit.fit.deviance,
        fit.fit.beta.to_vec(),
    )
}

fn assert_wiggle_engaged(link: &str, degree: Option<usize>, knots: Option<usize>, deviance: f64) {
    assert!(
        degree.is_some() && knots.is_some(),
        "flexible({link}): the monotone warp must be engaged (wiggle_degree/wiggle_knots present), \
         got degree={degree:?} knots={knots:?} — a silent drop to the fixed base link would leave \
         these None"
    );
    assert!(
        deviance.is_finite(),
        "flexible({link}): deviance must be finite, got {deviance}"
    );
}

#[test]
fn flexible_cloglog_control_binomial_wiggle_converges_2155() {
    let (converged, degree, knots, dev, _) = fit_flexible("cloglog");
    assert!(converged, "flexible(cloglog) control must converge");
    assert_wiggle_engaged("cloglog", degree, knots, dev);
}

#[test]
fn flexible_loglog_binomial_wiggle_converges_2155() {
    let (converged, degree, knots, dev, _) = fit_flexible("loglog");
    assert!(converged, "flexible(loglog) must converge (#2155)");
    assert_wiggle_engaged("loglog", degree, knots, dev);
}

#[test]
fn flexible_cauchit_binomial_wiggle_converges_2155() {
    let (converged, degree, knots, dev, _) = fit_flexible("cauchit");
    assert!(converged, "flexible(cauchit) must converge (#2155)");
    assert_wiggle_engaged("cauchit", degree, knots, dev);
}

/// The three links, fitted on identical data, must yield genuinely distinct
/// coefficient vectors — a gate re-restriction (fit fails) or a silent fallback
/// to a single fixed base link (identical fits) is caught here.
#[test]
fn flexible_loglog_cauchit_are_distinct_from_cloglog_2155() {
    let (_, _, _, _, cloglog) = fit_flexible("cloglog");
    let (_, _, _, _, loglog) = fit_flexible("loglog");
    let (_, _, _, _, cauchit) = fit_flexible("cauchit");

    let max_abs_diff = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len(), "coefficient vectors must share a width");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    let d_loglog = max_abs_diff(&loglog, &cloglog);
    let d_cauchit = max_abs_diff(&cauchit, &cloglog);
    eprintln!("[#2155] warped-fit max|Δβ|: loglog-vs-cloglog={d_loglog:.4}, cauchit-vs-cloglog={d_cauchit:.4}");
    assert!(
        d_loglog > 1.0e-4,
        "flexible(loglog) and flexible(cloglog) fits are indistinguishable ({d_loglog:.2e}) — \
         loglog likely fell back to a fixed base link"
    );
    assert!(
        d_cauchit > 1.0e-4,
        "flexible(cauchit) and flexible(cloglog) fits are indistinguishable ({d_cauchit:.2e})"
    );
}
