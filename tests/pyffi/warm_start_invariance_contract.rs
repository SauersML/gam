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
//! (families × constraint shapes) each run twice in one process — first
//! against a guaranteed-cold persistent warm-start store, then against the
//! now-warm store — asserting criterion-value and coefficient agreement.
//!
//! ## Mechanics
//!
//! The persistent warm-start store is anchored under
//! `std::env::temp_dir()/gam/warm/v1` (`src/solver/persistent_warm_start.rs`,
//! `persistent_store`), and on Unix `temp_dir()` resolves `TMPDIR`. Before
//! each configuration's FIRST fit we point `TMPDIR` at a fresh, empty,
//! per-run directory, so that fit is cold by construction; the second fit
//! of the same configuration then sees the warm entries the first one
//! wrote (including the data-independent seed-prefix payload that handed
//! #873 its divergent rho seed). Re-pointing per configuration also keeps
//! one configuration's seed-prefix entries from pre-warming the next
//! configuration's "cold" arm.
//!
//! ## The contract
//!
//! For every configuration, cold and warm must BOTH converge, and must
//! agree on the outer criterion value (REML/LAML) and on every
//! coefficient. The tolerances are far above two-paths-to-one-optimum
//! noise and far below any genuine alternate-stationary-point divergence
//! (#873's was ~30% of the curve's range). When a future cache key goes
//! stale-blind again, this fails in CI instead of in a user's pipeline.

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

/// Relative tolerance for cold-vs-warm agreement of the outer criterion
/// value. Two solves of the same problem from different starts agree to
/// roughly solver tolerance at the shared optimum; a warm seed landing on
/// a DIFFERENT stationary point moves the criterion by orders of
/// magnitude more than this.
const CRITERION_REL_TOL: f64 = 1e-6;

/// Sup-norm tolerance for cold-vs-warm coefficient agreement, relative to
/// the coefficient scale. Looser than the criterion bound because nearly
/// flat ρ directions can move β legitimately harder than the criterion,
/// but still ~two orders below the #873-class divergence.
const COEF_REL_TOL: f64 = 1e-4;

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

fn fit_once(case: &InvarianceCase, x: &[f64], y: &[f64], arm: &str) -> (f64, Vec<f64>) {
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
    std::fs::remove_file(&tmp).ok();

    let cfg = FitConfig {
        family: Some(case.family.to_string()),
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
        fit.fit.reml_score.is_finite(),
        "[{}/{}] non-finite criterion value",
        case.name,
        arm
    );
    (fit.fit.reml_score, fit.fit.beta.to_vec())
}

fn salt() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
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
            name: "gaussian_monotone",
            family: "gaussian",
            formula: "y ~ s(x, k=10, shape=monotone_increasing)",
            data: monotone_constrained_data,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for (idx, case) in cases.iter().enumerate() {
        // Fresh private store per configuration: the first arm is cold by
        // construction, AND the previous configuration's data-independent
        // seed-prefix entries cannot pre-warm this one.
        let mut store_root = std::env::temp_dir();
        store_root.push(format!(
            "gam_wsi_cold_{}_{}_{}",
            std::process::id(),
            idx,
            salt()
        ));
        std::fs::create_dir_all(&store_root).expect("create private cold-store TMPDIR");
        // SAFETY: single #[test] binary, set between fits on the test
        // thread before the next fit reads the environment. Edition-2024
        // marks `set_var` unsafe.
        unsafe {
            std::env::set_var("TMPDIR", &store_root);
        }
        assert_eq!(
            std::env::temp_dir(),
            store_root,
            "TMPDIR override did not take effect; cannot guarantee a cold cache"
        );

        let (x, y) = (case.data)();
        let (crit_cold, beta_cold) = fit_once(case, &x, &y, "cold");
        let (crit_warm, beta_warm) = fit_once(case, &x, &y, "warm");

        let crit_diff = (crit_cold - crit_warm).abs();
        let crit_tol = CRITERION_REL_TOL * crit_cold.abs().max(1.0);
        if crit_diff > crit_tol {
            failures.push(format!(
                "[{}] criterion depends on cache state: cold={:.10e} warm={:.10e} \
                 |Δ|={:.3e} > tol={:.3e}",
                case.name, crit_cold, crit_warm, crit_diff, crit_tol
            ));
        }

        if beta_cold.len() != beta_warm.len() {
            failures.push(format!(
                "[{}] coefficient layout depends on cache state: cold p={} warm p={}",
                case.name,
                beta_cold.len(),
                beta_warm.len()
            ));
        } else {
            let beta_scale = beta_cold
                .iter()
                .fold(0.0f64, |acc, b| acc.max(b.abs()))
                .max(1.0);
            let max_diff = beta_cold
                .iter()
                .zip(beta_warm.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let beta_tol = COEF_REL_TOL * beta_scale;
            if max_diff > beta_tol {
                failures.push(format!(
                    "[{}] coefficients depend on cache state: sup|Δβ|={:.3e} > tol={:.3e} \
                     (scale={:.3e})",
                    case.name, max_diff, beta_tol, beta_scale
                ));
            }
        }

        std::fs::remove_dir_all(&store_root).ok();
    }

    assert!(
        failures.is_empty(),
        "warm-start invariance contract violated — a cache must never change \
         the fitted result:\n{}",
        failures.join("\n")
    );
}
