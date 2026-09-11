//! Warm-start invariance contract (#969): warm state may change wall-clock
//! only, never the fixed point.
//!
//! ## Why this harness exists
//!
//! Every warm/cold divergence in the tracker (#873's cache-state-dependent
//! concave fit, #869's topology-blind cache key) was found by a user-level
//! symptom and fixed point-wise; nothing prevented the CLASS. A fit whose
//! result depends on what you ran before is the worst bug genus a
//! statistics tool can have, because it is invisible to any single-run
//! test. This harness is the permanent regression net: a matrix of fits
//! (families × constraint shapes) each run through a cold fit, a cache-priming
//! fit, and a warm fit in one process, asserting criterion-value and
//! coefficient agreement.
//!
//! ## Mechanics
//!
//! Every case owns an empty explicit store root. The cold arm carries no store;
//! a priming arm writes through the configured capability, and the warm arm
//! repeats with a clone of that same handle. No process history or ambient temp
//! setting can enter the comparison.
//!
//! ## The contract, and why it is not bitwise
//!
//! Both arms converging is not a separate assertion — a fit object only ever
//! comes from a converged optimization, so an arm that fails to certify fails
//! the fit call and names itself.
//!
//! Bit-identity was the obvious contract to want here and it was tried: the
//! criterion, β and the certified log-λ were compared with `to_bits()`. It is
//! not achievable, and the measurement says exactly why. On `gaussian_smooth`
//! (a family with no estimated nuisance at all, so nothing to do with #2363):
//!
//! ```text
//! criterion     cold=-1.011582068973655e3  warm=-1.011582068973655e3  2 ulps
//! β[1]          cold= 7.892570385059583e-1 warm= 7.892570383686568e-1  |Δ|=1.4e-10
//! certified ρ̂   cold=-2.230626112156725e0  warm=-2.230625914619381e0  |Δ|=2.0e-7
//! ```
//!
//! That is not a second optimum, it is one optimum located twice. The outer
//! search certifies `‖P∇F‖ ≤ 1e-7`; a certificate is a tolerance, not an
//! equation, so two arms entering from different seeds stop at two different
//! points inside the same tolerance ball. The criterion moving by 2 ulps while
//! ρ̂ moves by 2e-7 is the *proof* of that: `F` is stationary at the optimum, so
//! a within-ball ρ̂ displacement can only move `F` to second order. A cache that
//! donates a starting point therefore cannot produce bit-identical iterates
//! unless the terminal point is itself canonicalised, which is a different
//! design (and a much more expensive one) than this issue's.
//!
//! Every bound below is far tighter than what this gate allowed before — but
//! they are NOT all the same kind of number, and the difference matters to
//! anyone who later wants to tighten one:
//!
//! | quantity | old bound | bound here | kind | measured | #2363 defect |
//! |---|---|---|---|---|---|
//! | criterion | 1e-6 relative | 1024 ulps ≈ 2e-13 relative | **DERIVED** | 2 ulps | ~4e11 ulps |
//! | β | 1e-4 relative | 1e-7 relative | **GUARD** | 1.4e-10 | 4.98e-4 |
//! | certified ρ̂ | *not checked* | 1e-5 absolute | **GUARD** | 2.0e-7 | — |
//!
//! **DERIVED** means the number follows from the structure of the problem and
//! tightening it would contradict an argument. The criterion bound is the only
//! one of those, and it is the load-bearing assertion: at a certified
//! stationary point the criterion is flat in ρ, so cold-vs-warm agreement is
//! limited only by floating-point reassociation over the O(n) reduction plus a
//! second-order term — a handful of ulps, measured at 2. The #2363 defect sat
//! nine orders above it.
//!
//! **GUARD** means a coarse bound with deliberate slack, calibrated to the
//! measurement rather than deduced. Both size the certified-optimum ball,
//! `‖Δρ̂‖ ≲ ‖H⁻¹‖·‖P∇F‖`, whose amplification along a flat REML direction is
//! problem-dependent — there is no a-priori constant to compute, so these carry
//! the measured column as their only justification. The ρ̂ guard in particular
//! sits about two orders above what was measured. That slack is intentional
//! headroom for a legitimately flatter fixture, not a derivation, and it can be
//! tightened freely on evidence: doing so contradicts nothing. Both still sit
//! four orders below a genuine basin change (#873's was ~30% of the ρ range).
//!
//! Bit-identity IS asserted, but on the thing that is exactly reproducible by
//! construction rather than on an optimizer's terminal iterate: the frozen
//! nuisance itself, in
//! `lambda_search_nuisance_freeze_is_a_function_of_data_and_spec_alone_2363`
//! (`gam-solve --lib`). That is the quantity #2363 was actually about, it is a
//! deterministic sub-computation rather than a search result, and it is
//! bit-identical across donated warm βs and heuristic λs.
//!
//! Failures name the first offending coordinate and its ulp distance, so a
//! regression reports how far it moved rather than only that it moved.

use gam::inference::data::load_csvwith_inferred_schema;
use gam::init_parallelism;
use gam::solver::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use std::io::Write;

struct InvarianceCase {
    name: &'static str,
    family: &'static str,
    formula: &'static str,
    /// (x, y) generator; deterministic so both arms see identical data.
    data: fn() -> (Vec<f64>, Vec<f64>),
}

/// Everything an arm has to reproduce exactly.
///
/// There is no `converged` field because there is nothing to compare: a fit
/// object only ever comes from a converged optimization, so an arm that
/// reaches here at all has certified, and one that did not would have failed
/// the `fit_from_formula` unwrap in `fit_once` naming which arm it was.
struct ArmOutcome {
    criterion: f64,
    beta: Vec<f64>,
    log_lambdas: Vec<f64>,
}

/// Signed ulp distance between two `f64`s, using the standard
/// monotone-ordering trick so the count stays meaningful across zero.
fn ulp_distance(left: f64, right: f64) -> i128 {
    let order = |value: f64| -> i128 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            (i64::MIN - bits) as i128
        } else {
            bits as i128
        }
    };
    order(left) - order(right)
}

/// DERIVED. Ulp budget for the outer criterion — the load-bearing assertion of
/// this gate. At a certified stationary point `F` is flat in ρ, so two arms
/// that stop at different points inside the same tolerance ball can differ only
/// by floating-point reassociation over the O(n) reduction plus a second-order
/// term. Measured at 2 ulps; this leaves three decimal digits of headroom and
/// still sits nine orders below the #2363 defect (~4e11 ulps).
///
/// Tightening this contradicts the derivation above unless the reduction's
/// reassociation behaviour changes with it.
const CRITERION_ULP_BUDGET: i128 = 1024;

/// GUARD, not a derivation. Sup-norm bound on the certified log-λ. It sizes the
/// certified-optimum ball, `‖Δρ̂‖ ≲ ‖H⁻¹‖·‖P∇F‖`, with the outer search
/// certifying `‖P∇F‖ ≤ 1e-7` — but the flat-direction amplification `‖H⁻¹‖` is
/// problem-dependent and is not computed here, so there is no constant to
/// deduce. The only justification is the measured 2.0e-7, about two orders
/// below this bound; the slack is deliberate headroom for a legitimately
/// flatter fixture. Tighten on evidence whenever you like — nothing here
/// argues for 1e-5 specifically.
const RHO_ABS_BOUND: f64 = 1e-5;

/// GUARD, not a derivation. Sup-norm bound on β relative to its own scale. β̂ is
/// the inner mode at ρ̂ and `ρ ↦ β̂(ρ)` is Lipschitz, so a within-ball ρ̂
/// displacement moves β by a comparable amount — but the Lipschitz constant is
/// not computed, so this is calibrated to the measured 1.4e-10 rather than
/// deduced from it. Still a thousand times tighter than the 1e-4 this gate
/// allowed before, and far below the 4.98e-4 the #2363 Beta defect produced.
const COEF_REL_BOUND: f64 = 1e-7;

/// Describe the first coordinate at which two arms disagree by more than
/// `budget_ulps`, or `None` when every coordinate is within it.
fn first_ulp_gap(cold: &[f64], warm: &[f64], budget_ulps: i128) -> Option<String> {
    describe_first_gap(cold, warm, |a, b| ulp_distance(a, b).abs() > budget_ulps)
}

/// Describe the first coordinate at which two arms disagree by more than
/// `bound`, or `None` when every coordinate is within it.
fn first_absolute_gap(cold: &[f64], warm: &[f64], bound: f64) -> Option<String> {
    describe_first_gap(cold, warm, move |a, b| (a - b).abs() > bound)
}

fn describe_first_gap(
    cold: &[f64],
    warm: &[f64],
    exceeds: impl Fn(f64, f64) -> bool,
) -> Option<String> {
    if cold.len() != warm.len() {
        return Some(format!(
            "layout differs: cold len={} warm len={}",
            cold.len(),
            warm.len()
        ));
    }
    cold.iter()
        .zip(warm.iter())
        .enumerate()
        .find(|(_, (a, b))| exceeds(**a, **b))
        .map(|(index, (a, b))| {
            format!(
                "coordinate {index}: cold={a:.17e} warm={b:.17e} ulps={} |Δ|={:.3e}",
                ulp_distance(*a, *b),
                (a - b).abs()
            )
        })
}

/// Deterministic 32-bit LCG (Numerical Recipes constants) for
/// reproducible jitter/thresholds without a rand dependency.
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
    (*state as f64) / 4294967296.0
}

fn gaussian_smooth_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x9e3779b9u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let noise = 0.1 * (lcg_uniform(&mut rng) - 0.5);
            (
                x,
                (2.0 * std::f64::consts::PI * x).sin() + 0.5 * (5.0 * x).cos() + noise,
            )
        })
        .unzip()
}

fn poisson_count_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x6a09e667u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let lambda = (0.5 + (2.0 * std::f64::consts::PI * x).sin()).exp();
            // Deterministic count draw: invert a uniform through a crude
            // discretization (lambda + centered jitter, floored at 0).
            let y = (lambda + 2.0 * (lcg_uniform(&mut rng) - 0.5))
                .round()
                .max(0.0);
            (x, y)
        })
        .unzip()
}

fn binomial_data() -> (Vec<f64>, Vec<f64>) {
    let n = 600usize;
    let mut rng = 0xb7e15162u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let eta = 2.0 * (2.0 * std::f64::consts::PI * x).sin();
            let p = 1.0 / (1.0 + (-eta).exp());
            let y = if lcg_uniform(&mut rng) < p { 1.0 } else { 0.0 };
            (x, y)
        })
        .unzip()
}

fn negative_binomial_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x243f6a88u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.2 + 1.5 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            // A deterministic overdispersed count fixture. Its non-flat
            // conditional mean makes seed-η and converged-η profiling
            // materially different, directly exercising #2363.
            let multiplier = if lcg_uniform(&mut rng) < 0.3 {
                0.15
            } else {
                1.75
            };
            let y = (mu * multiplier + lcg_uniform(&mut rng)).round().max(0.0);
            (x, y)
        })
        .unzip()
}

fn gamma_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x13198a2eu64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.3 + 0.9 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            let y = mu * (0.7 + 0.6 * lcg_uniform(&mut rng));
            (x, y)
        })
        .unzip()
}

fn tweedie_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0xa4093822u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.1 + 1.1 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            let u = lcg_uniform(&mut rng);
            let y = if u < 0.2 {
                0.0
            } else {
                mu * (0.55 + 0.9 * lcg_uniform(&mut rng))
            };
            (x, y)
        })
        .unzip()
}

fn beta_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x082efa98u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let eta = 1.8 * (2.0 * std::f64::consts::PI * x).sin();
            let mu = 1.0 / (1.0 + (-eta).exp());
            let y = (mu + 0.18 * (lcg_uniform(&mut rng) - 0.5)).clamp(0.01, 0.99);
            (x, y)
        })
        .unzip()
}

fn monotone_constrained_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x3c6ef372u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            // Monotone trend with wiggle the constraint must fight, plus
            // jitter — the binding-constraint shape whose cold seeds the
            // #509/#873 class rejected.
            let noise = 0.05 * (lcg_uniform(&mut rng) - 0.5);
            (x, x + 0.08 * (4.0 * std::f64::consts::PI * x).sin() + noise)
        })
        .unzip()
}

fn fit_once(
    case: &InvarianceCase,
    x: &[f64],
    y: &[f64],
    arm: &str,
    persistent_warm_start_store: Option<gam::warm_start::ConfiguredWarmStartStore>,
) -> ArmOutcome {
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_wsi_{}_{}_{}.csv",
        case.name,
        arm,
        std::process::id()
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic data");
    if let Err(err) = std::fs::remove_file(&tmp) {
        eprintln!(
            "warning: could not remove temp csv {}: {err}",
            tmp.display()
        );
    }

    let cfg = FitConfig {
        family: Some(case.family.to_string()),
        persistent_warm_start_store,
        ..FitConfig::default()
    };
    let result = fit_from_formula(case.formula, &ds, &cfg).unwrap_or_else(|e| {
        panic!(
            "[{}/{}] fit aborted — a {} fit must succeed regardless of cache state: {e}",
            case.name, arm, arm
        )
    });
    let FitResult::Standard(fit) = result else {
        panic!("[{}/{}] expected a Standard GAM fit", case.name, arm);
    };
    assert!(
        fit.fit.reml_score().is_some_and(f64::is_finite),
        "[{}/{}] non-finite criterion value",
        case.name,
        arm
    );
    ArmOutcome {
        criterion: fit
            .fit
            .reml_score()
            .expect("the fit reports a REML/LAML criterion"),
        beta: fit.fit.beta.to_vec(),
        log_lambdas: fit.fit.log_lambdas.to_vec(),
    }
}

#[test]
fn fits_are_invariant_to_warm_start_cache_state_across_families() {
    init_parallelism();

    let cases = [
        InvarianceCase {
            name: "gaussian_smooth",
            family: "gaussian",
            formula: "y ~ s(x, k=10)",
            data: gaussian_smooth_data,
        },
        InvarianceCase {
            name: "poisson_smooth",
            family: "poisson",
            formula: "y ~ s(x, k=8)",
            data: poisson_count_data,
        },
        InvarianceCase {
            name: "binomial_smooth",
            family: "binomial",
            formula: "y ~ s(x, k=8)",
            data: binomial_data,
        },
        InvarianceCase {
            name: "negative_binomial_smooth",
            family: "negative-binomial",
            formula: "y ~ s(x, k=8)",
            data: negative_binomial_data,
        },
        InvarianceCase {
            name: "gamma_smooth",
            family: "gamma",
            formula: "y ~ s(x, k=8)",
            data: gamma_data,
        },
        InvarianceCase {
            name: "gamma_matern",
            family: "gamma",
            formula: "y ~ matern(x, k=8)",
            data: gamma_data,
        },
        InvarianceCase {
            name: "tweedie_smooth",
            // A Tweedie fit needs an explicit variance power (a893d85bc); the
            // cache-state invariance under test does not depend on its value.
            family: "tweedie(p=1.5)",
            formula: "y ~ s(x, k=8)",
            data: tweedie_data,
        },
        InvarianceCase {
            name: "beta_smooth",
            family: "beta",
            formula: "y ~ s(x, k=8)",
            data: beta_data,
        },
        InvarianceCase {
            name: "gaussian_monotone",
            family: "gaussian",
            formula: "y ~ s(x, k=10, shape=monotone_increasing)",
            data: monotone_constrained_data,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let (x, y) = (case.data)();
        let cache_directory = tempfile::tempdir().expect("create private warm-start root");
        let cache = FitConfig::default()
            .with_persistent_warm_start_root(cache_directory.path().join("warm"))
            .persistent_warm_start_store
            .expect("explicit root must configure a store");
        let cold = fit_once(case, &x, &y, "cold", None);
        // Prime both the exact-key and seed-prefix checkpoints. This arm need
        // not itself be cold; after it completes the next arm is guaranteed to
        // have a valid persistent entry.
        drop(fit_once(case, &x, &y, "prime", Some(cache.clone())));
        let warm = fit_once(case, &x, &y, "warm", Some(cache));

        if let Some(gap) = first_ulp_gap(
            std::slice::from_ref(&cold.criterion),
            std::slice::from_ref(&warm.criterion),
            CRITERION_ULP_BUDGET,
        ) {
            failures.push(format!(
                "[{}] criterion depends on cache state (budget {CRITERION_ULP_BUDGET} ulps): {gap}",
                case.name
            ));
        }
        let beta_scale = cold
            .beta
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()))
            .max(1.0);
        if let Some(gap) = first_absolute_gap(&cold.beta, &warm.beta, COEF_REL_BOUND * beta_scale) {
            failures.push(format!(
                "[{}] coefficients depend on cache state (bound {:.3e}): {gap}",
                case.name,
                COEF_REL_BOUND * beta_scale
            ));
        }
        if let Some(gap) = first_absolute_gap(&cold.log_lambdas, &warm.log_lambdas, RHO_ABS_BOUND) {
            failures.push(format!(
                "[{}] certified log-λ depends on cache state (bound {RHO_ABS_BOUND:.3e}): {gap}",
                case.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "warm-start invariance contract violated — a cache must never change \
         the fitted result:\n{}",
        failures.join("\n")
    );
}
