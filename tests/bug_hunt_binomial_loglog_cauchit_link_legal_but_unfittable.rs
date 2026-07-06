//! #2158 regression: the binomial `loglog` and `cauchit` links pass validation
//! (`link_legal_for_family`, the #2104 fix) and are advertised everywhere — the
//! parser accepts them, the kernel implements their inverse links, and
//! `gam predict` handles them — yet a *fit* that requested either link used to
//! abort deep in the solver:
//!
//! ```text
//! error: standard term fit failed: optimize_external_design requires a
//! supported standard GLM family/link; got Binomial LogLog. …
//! ```
//!
//! `resolve_external_family` (crates/gam-solve/src/estimate/external_options.rs)
//! enumerated only logit/probit/cloglog/SAS/Beta-Logistic for the Binomial
//! family, so loglog/cauchit fell to the `_ => false` arm — a missing allow-list
//! entry, not a missing implementation (both carry full 5-jet Fisher weights via
//! `fisher_weight_jet5`, exactly like probit/cloglog).
//!
//! The test drives the FULL `fit_from_formula` pipeline on a separable-ish
//! binomial dataset three ways:
//!   * `cloglog` — supported control; proves the harness/data are otherwise
//!     valid (passed before the fix too),
//!   * `loglog`, `cauchit` — each must now fit.
//!
//! Beyond "it fits", the test pins the *behaviour* from a second angle so a
//! silent fallback to logit (report success but ignore the requested link) can't
//! pass: the fitted model must record the exact requested link in its
//! `saved_link_state`, and predicting on a grid must yield valid, strictly
//! monotone-increasing probabilities that recover the planted signal. The three
//! links, fitted on identical data, must also produce genuinely distinct fitted
//! curves — a fallback that collapsed loglog/cauchit onto the logit/cloglog map
//! would fail this.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{LinkFunction, ResponseFamily};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

// --- deterministic splitmix64 RNG — no external deps -------------------------
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `n` rows of a single-covariate binomial dataset with `x ~ U(-1.6, 1.6)` and
/// `P(y=1|x) = logistic(1.6·x)` — a genuinely covariate-driven, monotone,
/// separable-ish signal that any reasonable binomial link can fit. The truth is
/// logistic, but the point of the test is only that loglog/cauchit *fit* it and
/// recover the monotone trend, not that they match the logit shape exactly.
fn binomial_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(seed);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -1.6 + 3.2 * rng.unit();
        let p = logistic(1.6 * x);
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![y.to_string(), x.to_string()]));
    }
    let headers = ["y", "x"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset")
}

fn binomial_cfg() -> FitConfig {
    FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    }
}

/// The `LinkFunction` the resolved fitted family must record for a given
/// requested link name, so a silent fallback to logit is caught.
fn expected_link_function(link: &str) -> LinkFunction {
    match link {
        "cloglog" => LinkFunction::CLogLog,
        "loglog" => LinkFunction::LogLog,
        "cauchit" => LinkFunction::Cauchit,
        other => panic!("unexpected link {other}"),
    }
}

/// Fit `y ~ x + link(type=<link>)` as a binomial GAM through the full formula
/// pipeline, assert the resolved link matches the request, then predict the
/// response-scale probabilities on `grid`. Returns the predicted probabilities.
fn fit_and_predict(link: &str, grid: &[f64]) -> Vec<f64> {
    let data = binomial_dataset(1200, 20258);
    let cfg = binomial_cfg();
    let formula = format!("y ~ x + link(type={link})");

    let FitResult::Standard(fit) = fit_from_formula(&formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("binomial {link} fit must succeed (#2158), got: {e:?}"))
    else {
        panic!("expected a standard binomial GAM fit for link={link}");
    };

    // Second-angle guard: the fitted model must actually carry the requested
    // link, not silently fall back to logit while reporting success. Standard
    // links carry no learnable state, so the resolved family (not the
    // `saved_link_state` cell, which is `Standard(None)` for every state-less
    // link) is the authoritative record of what was fitted.
    let family = fit
        .fit
        .likelihood_family
        .clone()
        .unwrap_or_else(|| panic!("link={link}: fitted model must carry a resolved family"));
    assert_eq!(
        family.response,
        ResponseFamily::Binomial,
        "link={link}: expected a Binomial fit"
    );
    let want = expected_link_function(link);
    let got = family.link.link_function();
    assert_eq!(
        got, want,
        "link={link}: fitted model records the wrong link ({got:?}) — a silent fallback would \
         pass 'it fits' but be caught here"
    );

    assert!(
        fit.fit.deviance.is_finite(),
        "link={link}: fitted deviance must be finite, got {}",
        fit.fit.deviance
    );

    // Predict response-scale probabilities on the covariate grid using the
    // model's own resolved link.
    let xcol = data
        .headers
        .iter()
        .position(|h| h == "x")
        .expect("x column");
    let hlen = data.headers.len();
    let m = grid.len();
    let mut design_grid = Array2::<f64>::zeros((m, hlen));
    for (r, &x) in grid.iter().enumerate() {
        design_grid[[r, xcol]] = x;
    }
    let design = build_term_collection_design(design_grid.view(), &fit.resolvedspec)
        .expect("rebuild binomial design at the prediction grid");
    let dense = design.design.to_dense();
    let offset = Array1::<f64>::zeros(m);
    predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .unwrap_or_else(|e| panic!("predict under link={link} must succeed, got {e:?}"))
        .mean
        .to_vec()
}

/// Predicted probabilities on a shared grid must be valid, finite, and strictly
/// increasing (the planted signal is monotone increasing in x).
fn assert_valid_monotone_probs(link: &str, probs: &[f64]) {
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            p.is_finite() && p > 0.0 && p < 1.0,
            "link={link}: predicted probability #{i} = {p} is not a valid probability in (0,1)"
        );
    }
    for w in probs.windows(2) {
        assert!(
            w[1] > w[0],
            "link={link}: predicted probabilities must be strictly increasing in x \
             (monotone planted signal); got {} then {}",
            w[0],
            w[1]
        );
    }
    // Recovery: the fit spans a wide probability range (it tracks the strong
    // covariate signal, not a flat collapse-to-mean).
    let lo = probs.first().copied().unwrap();
    let hi = probs.last().copied().unwrap();
    assert!(
        hi - lo > 0.5,
        "link={link}: fitted probability range {lo:.3}..{hi:.3} is too flat — the covariate \
         signal was not recovered"
    );
}

/// Grid spanning the covariate support.
fn prediction_grid() -> Vec<f64> {
    (0..25).map(|i| -1.5 + 3.0 * (i as f64) / 24.0).collect()
}

#[test]
fn binomial_cloglog_control_fits_and_recovers() {
    // The supported control: proves the harness/data/prediction plumbing is
    // valid independent of the #2158 fix. Passed before the fix too.
    let grid = prediction_grid();
    let probs = fit_and_predict("cloglog", &grid);
    assert_valid_monotone_probs("cloglog", &probs);
}

#[test]
fn binomial_loglog_link_is_fittable_2158() {
    let grid = prediction_grid();
    let probs = fit_and_predict("loglog", &grid);
    assert_valid_monotone_probs("loglog", &probs);
}

#[test]
fn binomial_cauchit_link_is_fittable_2158() {
    let grid = prediction_grid();
    let probs = fit_and_predict("cauchit", &grid);
    assert_valid_monotone_probs("cauchit", &probs);
}

/// The three links, fitted on identical data, must yield genuinely distinct
/// fitted curves — a silent fallback that collapsed loglog/cauchit onto the
/// logit/cloglog inverse-link map would make these coincide and is thereby
/// caught even if the link-state guard were somehow bypassed.
#[test]
fn binomial_loglog_cauchit_curves_are_distinct_from_cloglog_2158() {
    let grid = prediction_grid();
    let cloglog = fit_and_predict("cloglog", &grid);
    let loglog = fit_and_predict("loglog", &grid);
    let cauchit = fit_and_predict("cauchit", &grid);

    let max_abs_diff = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };

    let d_loglog = max_abs_diff(&loglog, &cloglog);
    let d_cauchit = max_abs_diff(&cauchit, &cloglog);
    let d_between = max_abs_diff(&loglog, &cauchit);
    eprintln!(
        "[#2158] fitted-curve max|Δp|: loglog-vs-cloglog={d_loglog:.4}, \
         cauchit-vs-cloglog={d_cauchit:.4}, loglog-vs-cauchit={d_between:.4}"
    );
    assert!(
        d_loglog > 1.0e-3,
        "loglog and cloglog fitted curves are indistinguishable ({d_loglog:.2e}) — \
         loglog likely silently fell back to another link"
    );
    assert!(
        d_cauchit > 1.0e-3,
        "cauchit and cloglog fitted curves are indistinguishable ({d_cauchit:.2e}) — \
         cauchit likely silently fell back to another link"
    );
    assert!(
        d_between > 1.0e-3,
        "loglog and cauchit fitted curves are indistinguishable ({d_between:.2e})"
    );
}
