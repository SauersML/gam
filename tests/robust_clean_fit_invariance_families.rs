//! CLEAN-FIT INVARIANCE — the zero-downside proof, one family per test.
//!
//! THE ZERO-DOWNSIDE PROPERTY. The family-general Jeffreys penalty
//! `Φ = ½ log|I_r(β)|` is self-limiting: its score is `O(1)` against the data's
//! `O(n)` Fisher information, so on any WELL-IDENTIFIED direction its only effect
//! is the `O(1/n)` Firth bias-reduction nudge — negligible at the sample sizes
//! here. It bites ONLY where `I(β)` is near-singular (a separating direction).
//! Therefore, on a well-identified fit with ample data and NO separation, turning
//! the robustness flag ON (`RobustIdentification::FirthOnly` — full identifiable-
//! span Jeffreys, NO orthogonalization design surgery) must leave the fit
//! essentially unchanged: coefficients, the additive predictor (η at the training
//! rows), the log-likelihood, and the effective degrees of freedom all match to a
//! tight tolerance.
//!
//! One test per family that routes the robustness flag through a *distinct*
//! solver path:
//!   * `gaussian`        — standard GAM REML path (no Jeffreys term assembled for
//!                         identity link; ON must be byte-clean against OFF up to
//!                         the explicit ρ-prior).
//!   * `binomial-logit`  — standard GAM path WITH the link-general Jeffreys
//!                         operator assembled (`reml_robust_jeffreys_link`).
//!   * `gamlss`          — Gaussian location-scale (two coupled linear predictors,
//!                         custom-family joint-Newton path).
//!   * `survival`        — lognormal AFT location-scale (custom-family path).
//!   * `multinomial`     — softmax GLM (see the ignored test for why the flag is
//!                         not yet threaded through its public entrypoint).
//!
//! The BMS-probit family has its own dedicated clean-fit gate in
//! `robust_clean_fit_invariance.rs` (it needs the custom `BernoulliMarginalSlope`
//! request, not a formula).
//!
//! TOLERANCE RATIONALE. Both arms run through `fit_from_formula`, whose ρ-prior
//! defaults to `Normal{mean:0, sd:3}` — NOT the `Flat` sentinel — so the
//! firth-general PC-hyperprior default (which fires only on `Flat`) is NOT
//! triggered on either arm. The ONLY ON-vs-OFF difference is therefore the
//! Jeffreys term itself. The coefficient delta is bounded at a few percent of the
//! coefficient scale: the Firth `O(1/n)` correction at these sample sizes is far
//! below that, while any genuine `O(1)` perturbation would blow through it.
//!
//! Deterministic: fixed-seed LCG, no time / unseeded RNG.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, RobustIdentification, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
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

fn base_cfg(family: &str, robust: RobustIdentification) -> FitConfig {
    // The formula path's default ρ-prior is `Normal{0,3}` (NOT the `Flat`
    // sentinel), so the firth-general PC-hyperprior default does not fire on
    // either arm — the only ON-vs-OFF difference is the Jeffreys term.
    FitConfig {
        family: Some(family.to_string()),
        robust_identification: robust,
        ..FitConfig::default()
    }
}

/// Run a STANDARD GAM formula fit (`FitResult::Standard`) and reduce it to a
/// `CleanFit`. The additive predictor is rebuilt from the resolved spec at the
/// training design rows — the same `apply(&beta)` path the quality tests use.
fn run_standard(
    formula: &str,
    family: &str,
    robust: RobustIdentification,
    data: &gam::data::EncodedDataset,
    train_grid: &Array2<f64>,
) -> CleanFit {
    let cfg = base_cfg(family, robust);
    let result = fit_from_formula(formula, data, &cfg)
        .unwrap_or_else(|e| panic!("clean {family} fit (robust={robust:?}) returned Err: {e}"));
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
        robust,
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
    robust: RobustIdentification,
    family: &str,
) -> CleanFit {
    let edf_total = edf.unwrap_or(f64::NAN);
    let all_finite = beta.iter().all(|v| v.is_finite()) && eta_train.iter().all(|v| v.is_finite());
    eprintln!(
        "[clean-invariance:{family}] robust={robust:?} ll={log_likelihood:.6e} edf={edf_total:.6} \
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

/// The shared assertion battery: ON must reproduce OFF to a tight tolerance on
/// every reported quantity, and both arms must converge to a finite estimate.
/// Returns `(max|Δβ|, max|Δη|, |Δℓ|, |Δedf|)` for the per-family report.
fn assert_zero_downside(family: &str, off: &CleanFit, on: &CleanFit) -> (f64, f64, f64, f64) {
    assert!(
        off.all_finite && on.all_finite,
        "[{family}] non-finite coefficients / predictor on a clean fit (OFF finite={}, ON \
         finite={})",
        off.all_finite, on.all_finite,
    );
    assert!(
        off.outer_converged && on.outer_converged,
        "[{family}] clean fit failed to converge (OFF conv={}, ON conv={}) — the cohort is \
         well-identified, so both arms must settle",
        off.outer_converged, on.outer_converged,
    );
    assert_eq!(
        off.beta.len(),
        on.beta.len(),
        "[{family}] coefficient vector length changed between OFF and ON",
    );
    assert_eq!(
        off.eta_train.len(),
        on.eta_train.len(),
        "[{family}] training predictor length changed between OFF and ON",
    );

    // (1) COEFFICIENTS — bounded to the self-limiting Firth O(1/n) scale.
    let max_dbeta = off
        .beta
        .iter()
        .zip(on.beta.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    let beta_scale = off.beta.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_dbeta < 5e-2 * beta_scale.max(1.0),
        "[{family}] full-span Jeffreys perturbed a clean fit's coefficients beyond the O(1/n) \
         Firth scale: max|Δβ|={max_dbeta:.3e} (β scale {beta_scale:.3e}); zero-downside violated",
    );

    // (2) ADDITIVE PREDICTOR at the training rows — the prediction-equivalence
    //     proxy. Predictions must match to a tight absolute tolerance.
    let max_deta = off
        .eta_train
        .iter()
        .zip(on.eta_train.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    let eta_scale = off.eta_train.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_deta < 5e-2 * eta_scale.max(1.0),
        "[{family}] full-span Jeffreys changed the clean-fit predictions: max|Δη|={max_deta:.3e} \
         (η scale {eta_scale:.3e})",
    );

    // (3) LOG-LIKELIHOOD (a function of η) essentially unchanged.
    let dll = (off.log_likelihood - on.log_likelihood).abs();
    let ll_scale = off.log_likelihood.abs().max(1.0);
    assert!(
        dll < 5e-3 * ll_scale,
        "[{family}] full-span Jeffreys changed the clean-fit log-likelihood: |Δℓ|={dll:.3e} \
         (scale {ll_scale:.3e})",
    );

    // (4) EFFECTIVE DEGREES OF FREEDOM — Jeffreys adds no spurious curvature on
    //     identified directions, so model complexity is unchanged.
    let dedf = (off.edf_total - on.edf_total).abs();
    assert!(
        dedf.is_finite() && dedf < 5e-2 * off.edf_total.abs().max(1.0),
        "[{family}] full-span Jeffreys changed the clean-fit effective DoF: |Δedf|={dedf:.3e} \
         (edf {:.4})",
        off.edf_total,
    );

    // Smoothing parameters land in the same place (reported, not asserted-hard:
    // λ can drift slightly without moving β/η/ℓ when a direction is flat).
    let max_drho = if off.log_lambdas.len() == on.log_lambdas.len() {
        off.log_lambdas
            .iter()
            .zip(on.log_lambdas.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
    } else {
        f64::NAN
    };
    eprintln!(
        "[clean-invariance:{family}] ON-vs-OFF max|Δβ|={max_dbeta:.3e} max|Δη|={max_deta:.3e} \
         |Δℓ|={dll:.3e} |Δedf|={dedf:.3e} max|Δlogλ|={max_drho:.3e}",
    );
    (max_dbeta, max_deta, dll, dedf)
}

// ───────────────────────── GAUSSIAN ─────────────────────────

/// Clean, well-identified Gaussian additive fit. For the identity link the
/// link-general Jeffreys term is NOT assembled, so `FirthOnly` (under the shared
/// explicit ρ-prior) must reproduce OFF essentially byte-for-byte — the
/// strongest form of zero-downside.
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
    let off = run_standard(formula, "gaussian", RobustIdentification::Off, &data, &tg);
    let on = run_standard(formula, "gaussian", RobustIdentification::FirthOnly, &data, &tg);
    assert_zero_downside("gaussian", &off, &on);
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
    let off = run_standard(formula, "binomial", RobustIdentification::Off, &data, &tg);
    let on = run_standard(formula, "binomial", RobustIdentification::FirthOnly, &data, &tg);
    assert_zero_downside("binomial-logit", &off, &on);
}

// ───────────────────────── GAMLSS (Gaussian location-scale) ─────────────────────────

/// Clean Gaussian location-scale (GAMLSS) fit: a smooth mean and a smooth
/// log-σ, both well-identified on ample data. This routes the robustness flag
/// through the custom-family joint-Newton path (two coupled predictors). The
/// Jeffreys term is scoped to the joint under-identified span, which is empty on
/// this well-identified cohort, so ON must reproduce OFF.
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

    let run = |robust: RobustIdentification| -> CleanFit {
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1 + s(x, bs='tp', k=8)".to_string()),
            robust_identification: robust,
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(x, bs='tp', k=10)", &data, &cfg)
            .unwrap_or_else(|e| panic!("clean gamlss fit (robust={robust:?}) returned Err: {e}"));
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
            robust,
            "gamlss",
        )
    };

    let off = run(RobustIdentification::Off);
    let on = run(RobustIdentification::FirthOnly);
    assert_zero_downside("gamlss", &off, &on);
}

// ───────────────────────── SURVIVAL (lognormal AFT location-scale) ─────────────────────────

/// Clean lognormal AFT survival fit with a parametric covariate and a smooth
/// covariate, light censoring, ample events — well-identified. Routes the
/// robustness flag through the survival custom-family path. ON must reproduce
/// OFF.
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

    let run = |robust: RobustIdentification| -> CleanFit {
        let cfg = FitConfig {
            robust_identification: robust,
            ..FitConfig::default()
        };
        let result =
            fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=6)"#, &data, &cfg)
                .unwrap_or_else(|e| {
                    panic!("clean survival fit (robust={robust:?}) returned Err: {e}")
                });
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
            robust,
            "survival",
        )
    };

    let off = run(RobustIdentification::Off);
    let on = run(RobustIdentification::FirthOnly);
    assert_zero_downside("survival", &off, &on);
}

// ───────────────────────── MULTINOMIAL (softmax) ─────────────────────────

/// Clean multinomial-logit (softmax) zero-downside gate.
///
/// IGNORED — BLOCKED ON BUILD. The public multinomial entrypoint
/// (`fit_penalized_multinomial` / `MultinomialFitInputs`) does NOT yet expose a
/// `robust_identification` field: it hard-codes `RhoPrior::Flat` and constructs
/// its `BlockwiseFitOptions` with the default (`Off`) robustness policy when it
/// routes through `fit_custom_family_with_rho_prior`. There is therefore no way
/// to request `FirthOnly` on a multinomial fit through the supported API, so an
/// ON-vs-OFF comparison cannot be authored without asserting falsely (both arms
/// would be byte-identical OFF runs, which would be a vacuous "pass").
///
/// UN-IGNORE WHEN: `MultinomialFitInputs` gains a `robust_identification` field
/// (or `fit_penalized_multinomial` threads the policy into its
/// `BlockwiseFitOptions`). At that point this body becomes: build a clean,
/// well-separated 3-class softmax cohort, fit OFF and ON, and assert the
/// coefficient / fitted-probability / edf deltas are within the same tight
/// `assert_zero_downside`-style band.
#[test]
#[ignore = "multinomial robustness flag not yet wired (no `robust_identification` on \
            MultinomialFitInputs); un-ignore when the policy is threaded — see doc comment above"]
fn clean_fit_invariance_multinomial_softmax() {
    init_parallelism();
    // Intentionally empty: see the doc comment. Filling this in before the flag
    // is threaded would compare two OFF runs and falsely report zero-downside.
    panic!("multinomial robustness flag not yet wired; this test must stay #[ignore]");
}
