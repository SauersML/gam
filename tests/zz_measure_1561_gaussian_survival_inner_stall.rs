//! zz_measure DIAGNOSTIC (#1561): black-box discriminator for the survival
//! location-scale / marginal-slope inner-Newton STALL cluster (Group-1 of the
//! survival GAM_ERROR triage).
//!
//! What it measures
//! ----------------
//! Three quality tests refuse with "custom-family inner solve did not converge
//! after N cycle(s); refusing to expose profile objective derivatives ...":
//! `quality_vs_gamlss_gaussian_survival_ls` (SurvivalLocationScaleFamily,
//! theta_dim=5, seed ρ=[ln(0.01),0,0,0,0], ~70 cycles),
//! `quality_vs_lifelines_cox_like_marginal` (marginal-slope, oversmoothed seed
//! carrying a 12.0 ceiling coordinate), and `coxph_frailty` (latent). The open
//! question is WHY the inner joint-Newton exhausts its stall-guard budget:
//!   (ii) CONDITIONING — the Jeffreys/Firth term arms at an ILL-CONDITIONED
//!        (wiggly / under-identified) operating point (jeffreys.rs conditioning
//!        gate ramps its weight to 1 as λ_min → 0), and the Φ-augmented,
//!        γ≥0-constrained inner system is too stiff to reach the tightened
//!        1e-11 derivative-quality KKT tolerance before the #979 flat-residual /
//!        linear-rate guard bails; OR
//!   (iii) a GLOBAL epsilon-progress wall — the solve stalls at every operating
//!        point regardless of conditioning (the #979/#2132/#2228 family).
//! The two point at different fixes (a c0272802b-style certificate extension /
//! seed conditioning vs. an inner-solver progress guarantee), and the dashboard
//! log carries only the terminal cycle count and rho_checkpoint — not the
//! per-cycle residual/objective/gate trace that would settle it.
//!
//! Why a black-box ladder (reachability caveat)
//! --------------------------------------------
//! The per-cycle joint-Newton internals (KKT residual, |Δobjective| vs the
//! 64·eps·(1+|obj|) floor, scalar_relerr, hpen_nullity, linearized_rel, the
//! Jeffreys conditioning-gate weight, active I-spline rows, which stall guard
//! fires) live in `gam-custom-family` as `pub(crate)` and are not reachable from
//! a top-level integration test; nor does the public `FitConfig` expose the
//! outer seed ρ vector. So this test drives the reachable proxies:
//!   * Ladder A — sweep `time_smooth_lambda` (the ONE seed coordinate `FitConfig`
//!     exposes; it seeds the time-warp ρ, the first coordinate the failing seed
//!     pins at ln(0.01)) from wiggly (0.01) to smooth (10). If wiggly seeds STALL
//!     and smooth seeds CONVERGE, that is the signature of (ii); if every rung
//!     stalls, that is (iii).
//!   * Ladder B — sweep the LOCATION smooth's `k` (basis dimension = flexibility
//!     = degree of under-identification) from tight (k=3) to loose (k=10). More
//!     flexibility ⇒ smaller λ_min ⇒ more Jeffreys arming; if stalls appear only
//!     at high k that again isolates (ii)-conditioning.
//! Neither ladder can force the EXACT failing seed, so a converging rung does not
//! by itself certify a fix — it localizes the mechanism for the measured A10
//! repro that carries the per-cycle hooks. All verdicts print under
//! `[zz1561:gsurv]` / `[zz1561:mslope]`; the test hard-asserts only that every
//! rung ran to a finite, non-panicking verdict (a still-descending or genuinely
//! non-converged fit is a legitimate, recorded outcome, never a panic).

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use std::path::Path;
use std::time::Instant;

/// Byte-identical reproduction of the `quality_vs_gamlss_gaussian_survival_ls`
/// synthetic fixture (n=200 Weibull-AFT with a smooth x-dependent scale
/// envelope), drawn from the same Numerical-Recipes 64-bit LCG (state=1234) so
/// gam sees the identical rows the quality test refuses on.
fn build_gaussian_survival_dataset(n: usize) -> EncodedDataset {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state: u64 = 1234;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let shape = 1.5_f64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = -2.0 + 4.0 * next_unit();
        let scale = (-0.5 + 0.3 * (two_pi * xi).sin()).exp();
        let scale_envelope = 1.0 + 0.4 * (two_pi * xi).cos();
        let u = next_unit().max(1e-300);
        let base = scale * (-u.ln()).powf(1.0 / shape);
        let median = scale * (std::f64::consts::LN_2).powf(1.0 / shape);
        let t = (median + (base - median) * scale_envelope).max(1e-6);
        let ev = if next_unit() < 0.7 { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{:.17e}", 0.0_f64),
            format!("{:.17e}", t),
            format!("{:.17e}", ev),
            format!("{:.17e}", xi),
        ]));
    }
    let headers = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode survival location-scale data")
}

/// Number of cycles the inner solve reported before refusing, parsed from the
/// "did not converge after N cycle(s)" clause of the refusal certificate.
fn parse_cycles(err: &str) -> Option<usize> {
    let anchor = err.find("did not converge after ")? + "did not converge after ".len();
    let rest = &err[anchor..];
    let end = rest.find(" cycle")?;
    rest[..end].trim().parse::<usize>().ok()
}

/// The `rho_checkpoint=[...]` vector carried in the refusal certificate (the
/// best iterate the outer search reached), returned verbatim including brackets.
fn parse_rho_checkpoint(err: &str) -> Option<String> {
    let anchor = err.find("rho_checkpoint=[")? + "rho_checkpoint=".len();
    let rest = &err[anchor..];
    let end = rest.find(']')?;
    Some(rest[..=end].to_string())
}

/// Coarse exit classification from the refusal text, so the ladder shows WHICH
/// failure mode each rung hit (the inner-Newton deriv refusal vs. the
/// constrained active-set QP vs. an outer non-stationary certificate).
fn classify_error(err: &str) -> &'static str {
    if err.contains("refusing to expose profile objective derivatives") {
        "INNER-STALL(deriv-refusal)"
    } else if err.contains("linear-constrained Newton active-set failed")
        || err.contains("joint constrained Newton QP failed")
    {
        "ACTIVE-SET-QP"
    } else if err.contains("did not certify a stationary optimum") {
        "OUTER-NOT-STATIONARY"
    } else if err.contains("no candidate seeds passed outer startup validation") {
        "SEED-SCREENING(no seed started)"
    } else {
        "OTHER"
    }
}

/// One-line verdict for a survival location-scale run: on success the CONVERGED
/// operating point (log-λ vector, outer iterations, REML score, |g|) — the smooth
/// or wiggly ρ the fit actually landed on; on refusal the exit class, cycle
/// count, and rho_checkpoint.
fn summarize_locscale(
    result: Result<FitResult, String>,
    wall_ms: f64,
) -> String {
    match result {
        Ok(FitResult::SurvivalLocationScale(fit)) => {
            let u = &fit.fit.fit;
            let log_lambdas: Vec<f64> = u.log_lambdas.iter().copied().collect();
            format!(
                "CONVERGED wall_ms={wall_ms:.0} outer_iter={} reml={:.4} |g|={} \
                 log_lambdas={:?}",
                u.outer_iterations,
                u.reml_score,
                u.outer_gradient_norm
                    .map(|g| format!("{g:.3e}"))
                    .unwrap_or_else(|| "none".to_string()),
                log_lambdas
                    .iter()
                    .map(|v| format!("{v:.3}"))
                    .collect::<Vec<_>>(),
            )
        }
        Ok(_) => format!("CONVERGED(other-variant) wall_ms={wall_ms:.0}"),
        Err(err) => format!(
            "REFUSED wall_ms={wall_ms:.0} class={} cycles={} rho_checkpoint={}",
            classify_error(&err),
            parse_cycles(&err)
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".to_string()),
            parse_rho_checkpoint(&err).unwrap_or_else(|| "?".to_string()),
        ),
    }
}

fn base_locscale_cfg(time_smooth_lambda: f64) -> FitConfig {
    FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("s(x, k=4)".to_string()),
        // Mirror the quality fixture's warp knobs (#721): 2 internal knots, a
        // bounded outer budget so a stalling rung fails FAST instead of grinding.
        time_num_internal_knots: 2,
        time_smooth_lambda,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    }
}

/// Run the gaussian-survival location-scale fit at a given location smooth `k`
/// (`k=6` reproduces the quality fixture). The refusal reason is captured as a
/// Debug-formatted string instead of `expect`-ing, so a refusal is a measured
/// datum, not a panic.
fn run_locscale_k(
    ds: &EncodedDataset,
    cfg: &FitConfig,
    location_k: usize,
) -> (f64, Result<FitResult, String>) {
    let formula = format!("Surv(entry, exit, event) ~ s(x, k={location_k})");
    let started = Instant::now();
    let result = fit_from_formula(&formula, ds, cfg).map_err(|e| format!("{e:?}"));
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    assert!(wall_ms.is_finite(), "wall time must be finite");
    (wall_ms, result)
}

/// LADDER A: sweep the time-warp seed λ from wiggly to smooth on the exact
/// gaussian-survival fixture. Discriminates (ii)-conditioning (wiggly rungs
/// stall, smooth rungs converge) from (iii)-global-wall (all rungs stall).
#[test]
fn zz1561_gsurv_time_lambda_seed_ladder() {
    init_parallelism();
    let ds = build_gaussian_survival_dataset(200);
    // ln from -4.61 (0.01, the failing seed's first coordinate) to +2.30 (10.0).
    let time_lambdas = [0.01_f64, 0.05, 0.25, 1.0, 4.0, 10.0];
    eprintln!(
        "[zz1561:gsurv] LADDER A — time_smooth_lambda seed sweep (gaussian survival LS, \
         n=200, location s(x,k=6), scale s(x,k=4)); discriminator: do wiggly seeds stall \
         while smooth seeds converge?"
    );
    let mut rungs_run = 0usize;
    for &tsl in &time_lambdas {
        let cfg = base_locscale_cfg(tsl);
        let (wall_ms, result) = run_locscale_k(&ds, &cfg, 6);
        eprintln!(
            "[zz1561:gsurv]   time_smooth_lambda={tsl:>6.3} (ln={:+.2}) :: {}",
            tsl.ln(),
            summarize_locscale(result, wall_ms)
        );
        rungs_run += 1;
    }
    assert_eq!(rungs_run, time_lambdas.len(), "every rung must run");
}

/// LADDER B: sweep the LOCATION smooth's flexibility `k` (basis dim = degree of
/// under-identification) at a fixed neutral time seed. More flexibility ⇒ smaller
/// penalized-Hessian λ_min ⇒ more Jeffreys arming; stalls appearing only at high
/// k isolate (ii)-conditioning.
#[test]
fn zz1561_gsurv_location_flexibility_ladder() {
    init_parallelism();
    let ds = build_gaussian_survival_dataset(200);
    let ks = [3usize, 4, 6, 8, 10];
    eprintln!(
        "[zz1561:gsurv] LADDER B — location smooth k sweep (time_smooth_lambda=1.0, \
         scale s(x,k=4)); discriminator: does under-identification (larger k) trigger the stall?"
    );
    let mut rungs_run = 0usize;
    for &k in &ks {
        let cfg = base_locscale_cfg(1.0);
        let (wall_ms, result) = run_locscale_k(&ds, &cfg, k);
        eprintln!(
            "[zz1561:gsurv]   location_k={k:>2} :: {}",
            summarize_locscale(result, wall_ms)
        );
        rungs_run += 1;
    }
    assert_eq!(rungs_run, ks.len(), "every rung must run");
}

/// SECOND ARM: survival marginal-slope (`quality_vs_lifelines_cox_like_marginal`
/// family). Its failing seed is OVERSMOOTHED (rho_checkpoint carries a 12.0
/// ceiling coordinate) with the `-d·log(qd1)` baseline-hazard barrier crawling
/// from a boundary cold start — the opposite end of the ladder from the wiggly
/// gaussian-survival seed. Sweeping `time_smooth_lambda` shows whether the stall
/// tracks the oversmoothed end (barrier / epsilon-progress, (iii)) or the wiggly
/// end. Fits the FULL heart-failure dataset (not the held-out train split of the
/// quality test) — a lighter reachable probe of the same inner solve.
#[test]
fn zz1561_marginal_slope_time_lambda_ladder() {
    init_parallelism();
    const HEART_CSV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/heart_failure_clinical_records_dataset.csv"
    );
    let ds = load_csvwith_inferred_schema(Path::new(HEART_CSV)).expect("load heart-failure csv");
    let time_lambdas = [0.01_f64, 0.05, 0.25, 1.0, 4.0, 10.0];
    eprintln!(
        "[zz1561:mslope] survival marginal-slope — time_smooth_lambda seed sweep on \
         heart-failure (full data, Surv(time, DEATH_EVENT) ~ sex + age, z=ejection_fraction); \
         discriminator: which end of the ladder stalls (oversmoothed barrier crawl vs wiggly)?"
    );
    let mut rungs_run = 0usize;
    for &tsl in &time_lambdas {
        let cfg = FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            z_column: Some("ejection_fraction".to_string()),
            time_smooth_lambda: tsl,
            outer_max_iter: Some(80),
            ..FitConfig::default()
        };
        let started = Instant::now();
        let result = fit_from_formula("Surv(time, DEATH_EVENT) ~ sex + age", &ds, &cfg)
            .map_err(|e| format!("{e:?}"));
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(wall_ms.is_finite(), "wall time must be finite");
        let verdict = match result {
            Ok(_) => format!("CONVERGED wall_ms={wall_ms:.0}"),
            Err(err) => format!(
                "REFUSED wall_ms={wall_ms:.0} class={} cycles={} rho_checkpoint={}",
                classify_error(&err),
                parse_cycles(&err)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                parse_rho_checkpoint(&err).unwrap_or_else(|| "?".to_string()),
            ),
        };
        eprintln!(
            "[zz1561:mslope]   time_smooth_lambda={tsl:>6.3} (ln={:+.2}) :: {verdict}",
            tsl.ln()
        );
        rungs_run += 1;
    }
    assert_eq!(rungs_run, time_lambdas.len(), "every rung must run");
}
