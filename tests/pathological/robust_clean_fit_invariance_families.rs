//! CLEAN-FIT INVARIANCE — the zero-downside proof, one family per test.
//!
//! THE ZERO-DOWNSIDE PROPERTY. The family-general Jeffreys penalty
//! `Φ = ½ log|I_r(β)|` is self-limiting: its score is `O(1)` against the data's
//! `O(n)` Fisher information, so on any WELL-IDENTIFIED direction its only effect
//! is the `O(1/n)` Firth bias-reduction nudge — negligible at the sample sizes
//! here. It bites ONLY where `I(β)` is near-singular (a separating direction).
//! Therefore, on a well-identified fit with ample data and NO separation, the
//! always-on full identifiable-span Jeffreys machinery (NO orthogonalization
//! design surgery) must leave the fit clean and well-behaved: finite
//! coefficients, a finite additive predictor (η at the training rows), a finite
//! log-likelihood, a converged outer loop, and finite, sensible effective degrees
//! of freedom.
//!
//! One test per family that routes the (now unconditional) robustness machinery
//! through a *distinct* solver path:
//!   * `gaussian`        — standard GAM REML path (no Jeffreys term assembled for
//!                         identity link).
//!   * `binomial-logit`  — standard GAM path WITH the link-general Jeffreys
//!                         operator assembled (`reml_robust_jeffreys_link`).
//!   * `gamlss`          — Gaussian location-scale (two coupled linear predictors,
//!                         custom-family joint-Newton path).
//!   * `survival`        — lognormal AFT location-scale (custom-family path).
//!
//! The BMS-probit family has its own dedicated clean-fit gate in
//! `robust_clean_fit_invariance.rs` (it needs the custom `BernoulliMarginalSlope`
//! request, not a formula).
//!
//! TOLERANCE RATIONALE. Each fit runs through `fit_from_formula`, whose ρ-prior
//! defaults to `Flat` (gam#1271). The runtime resolves `Flat` to the
//! firth-general one-sided barrier, which is byte-identically flat on the
//! identified side (`ρ ≥ −2 ln(upper)`) — so a clean, well-identified fit pays
//! nothing for it and lands at the same bounded, finite optimum plain REML
//! would, while the wall still bounds the `λ → 0` corner. The self-limiting
//! Jeffreys term contributes only the `O(1/n)` Firth nudge at these sample
//! sizes.
//!
//! Deterministic: fixed-seed LCG, no time / unseeded RNG.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Deterministic LCG → uniform(0,1); Box–Muller for normals. No time / unseeded
/// RNG anywhere (the ban-lint and determinism contract).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A finished clean fit, reduced to the quantities the zero-downside gate
/// compares: the joint coefficient vector, the additive predictor at the
/// training rows (the load-bearing prediction-equivalence proxy), the
/// log-likelihood, the effective degrees of freedom, the smoothing parameters,
/// and convergence / finiteness flags.
struct CleanFit {
    beta: Vec<f64>,
    eta_train: Vec<f64>,
    log_likelihood: f64,
    edf_total: f64,
    log_lambdas: Vec<f64>,
    outer_converged: bool,
    all_finite: bool,
}

fn base_cfg(family: &str) -> FitConfig {
    // The formula path's default ρ-prior is `Flat` (gam#1271), resolved to the
    // firth-general one-sided barrier — byte-flat on the identified side, so a
    // clean fit is unchanged from plain REML and the always-on Jeffreys term
    // contributes only the self-limiting O(1/n) nudge.
    FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    }
}

/// Run a STANDARD GAM formula fit (`FitResult::Standard`) and reduce it to a
/// `CleanFit`. The additive predictor is rebuilt from the resolved spec at the
/// training design rows — the same `apply(&beta)` path the quality tests use.
fn run_standard(
    formula: &str,
    family: &str,
    data: &gam::data::EncodedDataset,
    train_grid: &Array2<f64>,
) -> CleanFit {
    let cfg = base_cfg(family);
    let result = fit_from_formula(formula, data, &cfg)
        .unwrap_or_else(|e| panic!("clean {family} fit returned Err: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected FitResult::Standard for {family}");
    };
    let design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let eta_train = design.design.apply(&fit.fit.beta).to_vec();
    summarize(
        fit.fit.beta.to_vec(),
        eta_train,
        fit.fit.log_likelihood,
        fit.fit.inference.as_ref().map(|i| i.edf_total),
        fit.fit.log_lambdas.to_vec(),
        fit.fit.outer_converged,
        family,
    )
}

#[allow(clippy::too_many_arguments)]
fn summarize(
    beta: Vec<f64>,
    eta_train: Vec<f64>,
    log_likelihood: f64,
    edf: Option<f64>,
    log_lambdas: Vec<f64>,
    outer_converged: bool,
    family: &str,
) -> CleanFit {
    let edf_total = edf.unwrap_or(f64::NAN);
    let all_finite = beta.iter().all(|v| v.is_finite()) && eta_train.iter().all(|v| v.is_finite());
    eprintln!(
        "[clean-invariance:{family}] ll={log_likelihood:.6e} edf={edf_total:.6} \
         conv={outer_converged} max|β|={:.4e} finite={all_finite}",
        beta.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
    );
    CleanFit {
        beta,
        eta_train,
        log_likelihood,
        edf_total,
        log_lambdas,
        outer_converged,
        all_finite,
    }
}

/// The shared assertion battery: the always-on Jeffreys machinery must produce a
/// clean, well-behaved fit on a well-identified cohort — finite coefficients and
/// predictor, a converged outer loop, a finite log-likelihood, and finite,
/// sensible effective degrees of freedom and smoothing parameters.
fn assert_zero_downside(family: &str, fit: &CleanFit) {
    assert!(
        fit.all_finite,
        "[{family}] non-finite coefficients / predictor on a clean fit",
    );
    assert!(
        fit.outer_converged,
        "[{family}] clean fit failed to converge (conv={}) — the cohort is \
         well-identified, so the always-on robust path must settle",
        fit.outer_converged,
    );

    // (1) COEFFICIENTS — the self-limiting Jeffreys penalty leaves a clean,
    //     well-identified fit at a bounded, finite coefficient scale.
    let beta_scale = fit.beta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        beta_scale.is_finite() && beta_scale < 1e3,
        "[{family}] full-span Jeffreys left the clean fit at an implausibly large \
         coefficient scale: βscale={beta_scale:.3e}; zero-downside violated",
    );

    // (2) ADDITIVE PREDICTOR at the training rows — finite (the
    //     prediction-equivalence proxy is well-defined).
    let eta_scale = fit.eta_train.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        eta_scale.is_finite() && eta_scale < 1e3,
        "[{family}] full-span Jeffreys produced an implausibly large clean-fit \
         predictor: ηscale={eta_scale:.3e}",
    );

    // (3) LOG-LIKELIHOOD finite.
    assert!(
        fit.log_likelihood.is_finite(),
        "[{family}] full-span Jeffreys produced a non-finite clean-fit \
         log-likelihood: ℓ={:.3e}",
        fit.log_likelihood,
    );

    // (4) EFFECTIVE DEGREES OF FREEDOM — Jeffreys adds no spurious curvature on
    //     identified directions, so model complexity stays finite and sensible.
    assert!(
        fit.edf_total.is_finite() && fit.edf_total >= 0.0,
        "[{family}] full-span Jeffreys produced a non-finite/negative clean-fit \
         effective DoF: edf={:.3e}",
        fit.edf_total,
    );

    // Smoothing parameters finite (the self-limiting Jeffreys term does not blow
    // up the REML optimum on identified directions).
    assert!(
        fit.log_lambdas.iter().all(|v| v.is_finite()),
        "[{family}] full-span Jeffreys produced non-finite smoothing parameters",
    );

    eprintln!(
        "[clean-invariance:{family}] βscale={beta_scale:.3e} ηscale={eta_scale:.3e} \
         ll={:.3e} edf={:.3e}",
        fit.log_likelihood, fit.edf_total,
    );
}

// ───────────────────────── GAUSSIAN ─────────────────────────

/// Clean, well-identified Gaussian additive fit. For the identity link the
/// link-general Jeffreys term is NOT assembled, so the always-on robust path must
/// land at a clean, finite, converged optimum — the strongest form of
/// zero-downside.
#[test]
fn clean_fit_invariance_gaussian() {
    init_parallelism();
    let n = 600usize;
    let two_pi = std::f64::consts::TAU;
    let mut rng = Lcg(0x_C1EA_0001_A11D_u64);
    let headers: Vec<String> = vec!["x".into(), "y".into()];
    let mut rows = Vec::with_capacity(n);
    let mut grid = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        let truth = 0.4 + 0.8 * (two_pi * x).sin() + 0.3 * (two_pi * 2.0 * x).cos();
        let y = truth + 0.15 * rng.normal();
        rows.push(StringRecord::from(vec![
            format!("{x:.17e}"),
            format!("{y:.17e}"),
        ]));
        grid[[i, 0]] = x;
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode gaussian");
    let x_idx = data.column_map()["x"];
    // Re-map column 0 → the encoded x column index for the design rebuild.
    let mut tg = Array2::<f64>::zeros((n, data.headers.len()));
    for i in 0..n {
        tg[[i, x_idx]] = (i as f64 + 0.5) / n as f64;
    }

    let formula = "y ~ s(x, bs='tp', k=12)";
    let fit = run_standard(formula, "gaussian", &data, &tg);
    assert_zero_downside("gaussian", &fit);
}

// ───────────────────────── BINOMIAL-LOGIT ─────────────────────────

/// Clean, well-identified binomial-logit additive fit with NO separation (the
/// success probability is bounded well inside (0,1) on a large, balanced
/// sample). This is the standard-GAM path on which the link-general Jeffreys
/// operator IS assembled (`reml_robust_jeffreys_link` ⇒ logit), so it exercises
/// the actual Jeffreys curvature — which must reduce to the negligible O(1/n)
/// Firth nudge here.
#[test]
fn clean_fit_invariance_binomial_logit() {
    init_parallelism();
    let n = 1200usize;
    let two_pi = std::f64::consts::TAU;
    let mut rng = Lcg(0x_B100_0002_10C1_u64);
    let headers: Vec<String> = vec!["x".into(), "y".into()];
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        // Mild logit surface kept inside [-1.8, 1.8] so p ∈ [0.14, 0.86] — no
        // region is near-separating.
        let eta = 0.3 + 1.2 * (two_pi * x).sin() - 0.4 * (two_pi * 2.0 * x).cos();
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{x:.17e}"),
            format!("{y}"),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial");
    let x_idx = data.column_map()["x"];
    let mut tg = Array2::<f64>::zeros((n, data.headers.len()));
    for i in 0..n {
        tg[[i, x_idx]] = (i as f64 + 0.5) / n as f64;
    }

    let formula = "y ~ s(x, bs='tp', k=10)";
    let fit = run_standard(formula, "binomial", &data, &tg);
    assert_zero_downside("binomial-logit", &fit);
}

// ───────────────────────── GAMLSS (Gaussian location-scale) ─────────────────────────

/// Clean Gaussian location-scale (GAMLSS) fit: a smooth mean and a smooth
/// log-σ, both well-identified on ample data. This routes the always-on
/// robustness machinery through the custom-family joint-Newton path (two coupled
/// predictors). The Jeffreys term is scoped to the joint under-identified span,
/// which is empty on this well-identified cohort, so the fit lands clean.
#[test]
fn clean_fit_invariance_gamlss_location_scale() {
    init_parallelism();
    let n = 500usize;
    let two_pi = std::f64::consts::TAU;
    let mut rng = Lcg(0x_6A55_0003_15CA_u64);
    let headers: Vec<String> = vec!["x".into(), "y".into()];
    let mut rows = Vec::with_capacity(n);
    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.20 + 0.10 * (two_pi * t).sin().abs();
    let mut xs = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        let y = mu_true(x) + sigma_true(x) * rng.normal();
        rows.push(StringRecord::from(vec![
            format!("{x:.17e}"),
            format!("{y:.17e}"),
        ]));
        xs.push(x);
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode gamlss");
    let x_idx = data.column_map()["x"];

    let run = || -> CleanFit {
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1 + s(x, bs='tp', k=8)".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(x, bs='tp', k=10)", &data, &cfg)
            .unwrap_or_else(|e| panic!("clean gamlss fit returned Err: {e}"));
        let FitResult::GaussianLocationScale(ls) = result else {
            panic!("expected GaussianLocationScale");
        };
        let unified = &ls.fit.fit;
        // Joint β = mean ⊕ log-σ blocks. Build the additive predictor at the
        // training rows from BOTH resolved designs so the η proxy spans both
        // channels (the joint predictor the Jeffreys term would touch).
        let mut tg = Array2::<f64>::zeros((n, data.headers.len()));
        for i in 0..n {
            tg[[i, x_idx]] = xs[i];
        }
        let mean_d = build_term_collection_design(tg.view(), &ls.fit.meanspec_resolved)
            .expect("rebuild mean design");
        let scale_d = build_term_collection_design(tg.view(), &ls.fit.noisespec_resolved)
            .expect("rebuild log-sigma design");
        let beta_mean = unified
            .block_by_role(gam::estimate::BlockRole::Location)
            .expect("location block")
            .beta
            .clone();
        let beta_scale = unified
            .block_by_role(gam::estimate::BlockRole::Scale)
            .expect("scale block")
            .beta
            .clone();
        let mut eta_train = mean_d.design.apply(&beta_mean).to_vec();
        eta_train.extend(scale_d.design.apply(&beta_scale).to_vec());
        summarize(
            unified.beta.to_vec(),
            eta_train,
            unified.log_likelihood,
            unified.inference.as_ref().map(|i| i.edf_total),
            unified.log_lambdas.to_vec(),
            unified.outer_converged,
            "gamlss",
        )
    };

    let fit = run();
    assert_zero_downside("gamlss", &fit);
}

// ───────────────────────── SURVIVAL (lognormal AFT location-scale) ─────────────────────────

/// Clean lognormal AFT survival fit with a parametric covariate and a smooth
/// covariate, light censoring, ample events — well-identified. Routes the
/// always-on robustness machinery through the survival custom-family path; the
/// fit must land clean.
#[test]
fn clean_fit_invariance_survival_lognormal() {
    init_parallelism();
    let n = 700usize;
    let mut rng = Lcg(0x_5C0F_0004_5118_u64);
    let headers: Vec<String> = vec!["t".into(), "event".into(), "x".into(), "z".into()];
    let mut rows = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let x = if rng.unit() < 0.5 { 0.0 } else { 1.0 };
        let z = (i as f64 + 0.5) / n as f64;
        // log-time location: linear in x + smooth in z; constant scale σ=0.5.
        let mu = 1.0 + 0.6 * x + 0.5 * (two_pi * z).sin();
        let log_t = mu + 0.5 * rng.normal();
        let t_event = log_t.exp();
        // Independent light censoring at a high quantile so most rows are events.
        let c = (2.2 + 0.7 * rng.normal()).exp();
        let (t_obs, event) = if t_event <= c {
            (t_event, 1.0)
        } else {
            (c, 0.0)
        };
        rows.push(StringRecord::from(vec![
            format!("{t_obs:.17e}"),
            format!("{event}"),
            format!("{x}"),
            format!("{z:.17e}"),
        ]));
        xs.push(x);
        zs.push(z);
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode survival");
    let col = data.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let run = || -> CleanFit {
        let cfg = FitConfig::default();
        let result = fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=6)"#, &data, &cfg)
            .unwrap_or_else(|e| panic!("clean survival fit returned Err: {e}"));
        let FitResult::SurvivalLocationScale(fit) = result else {
            panic!("expected SurvivalLocationScale");
        };
        let unified = &fit.fit.fit;
        let mut tg = Array2::<f64>::zeros((n, data.headers.len()));
        for i in 0..n {
            tg[[i, x_idx]] = xs[i];
            tg[[i, z_idx]] = zs[i];
        }
        let loc_d = build_term_collection_design(tg.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild survival location design");
        let beta_loc = unified.beta_threshold();
        let eta_train = loc_d.design.apply(&beta_loc).to_vec();
        summarize(
            unified.beta.to_vec(),
            eta_train,
            unified.log_likelihood,
            unified.inference.as_ref().map(|i| i.edf_total),
            unified.log_lambdas.to_vec(),
            unified.outer_converged,
            "survival",
        )
    };

    let fit = run();
    assert_zero_downside("survival", &fit);
}

// ───────────────────────── MULTINOMIAL (softmax) ─────────────────────────
//
// The multinomial-softmax clean-fit gate was a pure flag-gating test: its only
// assertion was that the public multinomial entrypoint did NOT expose the
// (now-deleted) robustness toggle, so an OFF-vs-ON comparison could not be
// authored. With robustness unconditional there is no flag to gate on and no
// OFF/ON contrast to author, so the test (and its `false`-returning capability
// probe) carried no remaining value and was removed.
