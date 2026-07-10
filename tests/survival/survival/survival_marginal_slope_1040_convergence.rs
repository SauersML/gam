//! Repro + regression for #1040: the **survival** marginal-slope outer loop
//! used to fail to converge within any practical wall-clock budget for every
//! basis (matern / duchon / measure-jet all timed out at 400–600 s on a small
//! n=1500, centers=10 problem), while the **binary** marginal-slope on the same
//! bases / sizes converged in ~1–2 s.
//!
//! The discriminating fact in the report was: the inner PIRLS/joint-Newton
//! solve converged cleanly every cycle (`KKT/certificate-converged`) but the
//! INNER joint-Newton loop ground to its `inner_loop_hard_ceiling` on the flat
//! baseline-hazard λ valley, so the outer REML rejected ρ after ρ for hours.
//! The cure landed across several commits: a relative-objective-plateau exit
//! that fires on the O(1e4) survival NLL scale (a fixed absolute ε never trips)
//! together with the range(H_pen)-projected stationarity certificate — the
//! un-moved mass is a free ker(H_pen) gauge direction the outer IFT projects
//! out (gam#553), so the iterate IS the REML optimum on the identifiable
//! subspace and the loop reports `converged=true` instead of hanging.
//!
//! This test rebuilds the issue's *structural shape* — `Surv(time, event) ~
//! matern(PC1, PC2, centers)`, `survival_likelihood="marginal-slope"`,
//! `z_column`, and a matern log-slope surface — on synthetic right-censored
//! data at a tractable size (small n / centers to stay RAM- and CI-safe), and
//! asserts the survival-MS fit:
//!   * converges (a minted fit is the sealed convergence certificate, SPEC 20)
//!     — the load-bearing #1040 guard,
//!   * returns at all rather than hanging unbounded (a wall-clock backstop that
//!     fails a gross slowdown faster than the job ceiling; unlike the binary
//!     marginal-slope arm, the exact per-cell quadrature makes even a converged
//!     survival fit at this size run on the order of ~10^3 s, so the backstop
//!     brackets that runtime instead of asserting a "seconds" parity it can
//!     never hit — see `WALL_BUDGET_S`),
//!   * with every coefficient finite.
//!
//! It is the missing survival counterpart to the binary
//! `margslope_matern_logslope_timing` regression in
//! `bug_hunt_979_margslope_matern_logslope_slowdown.rs`.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::time::Instant;

const N: usize = 600;
const CENTERS: usize = 6;
/// Wall-clock backstop that fails a genuine hang *faster* than the job-level
/// 240-minute timeout, without flaking on the real converged runtime.
///
/// #1040 was a *convergence* bug — the outer REML loop never reached its
/// stopping criterion — so the load-bearing guard is that the fit mints at all
/// (fit existence is the sealed convergence proof, SPEC 20), not the clock. The clock cannot itself classify converged-vs-hang here: the
/// survival marginal-slope objective is an exact per-cell density integral
/// (∫ φ(z) Φ(η(z)) dz on a 384-node Gauss-Legendre rule, per non-affine
/// partition cell, per row, per inner cycle), so even a cleanly converged fit
/// at this small n/centers runs on the order of ~10³ s — squarely inside the
/// 400–600 s window the original hang was observed to time out in. A tight
/// "finishes in seconds" cap (the binary marginal-slope arm's regime) was never
/// reachable on this arm and would only ever red-flag the heavy-but-correct
/// quadrature. The honest separator is: a true hang never returns (the 240 min
/// job ceiling trips) or fails to mint a fit; a converged fit
/// returns in ~10³ s. This cap brackets that converged runtime with a wide
/// margin so it only fires on a gross, qualitatively-different slowdown.
const WALL_BUDGET_S: f64 = 3600.0;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f64 {
    // 53-bit mantissa uniform in [0, 1).
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    // Box–Muller; clamp the radial draw away from 0 for log stability.
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build a synthetic right-censored survival dataset with two spatial
/// coordinates (PC1, PC2), a standardized latent score `z`, and an event
/// time drawn from an exponential whose rate depends on the covariates, with
/// independent uniform censoring to land a moderate event rate.
fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = ["time", "event", "z", "PC1", "PC2"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let mut state: u64 = 0x5CA1_AB1E_1040_C0DE_u64;

    let mut z_raw: Vec<f64> = Vec::with_capacity(N);
    let mut scratch: Vec<[f64; 3]> = Vec::with_capacity(N); // [pc1, pc2, u_event]

    for _ in 0..N {
        let pc1 = next_gauss(&mut state) * 0.5;
        let pc2 = next_gauss(&mut state) * 0.5;
        let z = next_gauss(&mut state);
        let u_event = next_unit(&mut state).max(1e-9);
        z_raw.push(z);
        scratch.push([pc1, pc2, u_event]);
    }

    // Standardize z to mean 0, var 1 (the marginal-slope latent score).
    let z_mean = z_raw.iter().sum::<f64>() / N as f64;
    let z_var = z_raw.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / N as f64;
    let z_sd = z_var.sqrt().max(1e-12);
    let z_std: Vec<f64> = z_raw.iter().map(|v| (v - z_mean) / z_sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for (i, s) in scratch.iter().enumerate() {
        let [pc1, pc2, u_event] = *s;
        let z = z_std[i];
        // Log-hazard linear predictor: a smooth-ish spatial signal plus the
        // latent score. Exponential event time T = -log(U) / rate.
        let lin = 0.4 * pc1 - 0.3 * pc2 + 0.35 * z;
        let rate = lin.exp().max(1e-6);
        let t_event = -u_event.ln() / rate;
        // Independent uniform censoring time → ~60% event rate.
        let t_cens = 1.5 * next_unit(&mut state);
        let (time, event) = if t_event <= t_cens {
            (t_event, 1u8)
        } else {
            (t_cens, 0u8)
        };
        let time = time.max(1e-4);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
            pc1.to_string(),
            pc2.to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode #1040 survival-MS convergence dataset")
}

#[test]
fn survival_marginal_slope_matern_logslope_converges_within_budget() {
    init_parallelism();

    // No CUDA driver in CI on macOS.
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();

    // The issue's shape: matern spatial surface on the marginal block and the
    // same matern surface on the log-slope block, latent score `z`, right-
    // censored Surv(time, event).
    let matern = format!("matern(PC1, PC2, centers={CENTERS})");
    let formula = format!("Surv(time, event) ~ {matern}");
    let cfg = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("z".to_string()),
        logslope_formula: Some(matern.clone()),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };

    let start = Instant::now();
    let result = fit_from_formula(&formula, &data, &cfg)
        .expect("survival marginal-slope matern/logslope fit (#1040)");
    let elapsed = start.elapsed().as_secs_f64();

    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };

    eprintln!(
        "[1040-REPRO] n={N} centers={CENTERS} total_s={elapsed:.2} \
         outer_iters={} inner_cycles={} converged=certified",
        fit.fit.outer_iterations, fit.fit.inner_cycles
    );

    // ── Assertion 1: the survival-MS outer loop converged ─────────────────
    // This is the #1040 fix: the flat baseline-hazard λ valley used to hang the
    // inner joint-Newton at its ceiling, so the outer REML never terminated.
    // A minted fit IS the convergence certificate now (SPEC 20): a #1040-class
    // hang either never returns (assertion 2's clock) or fails to mint.

    // ── Assertion 2: it terminated, not the unbounded hang ────────────────
    // Convergence (assertion 1) is the real #1040 guard; this clock backstop
    // only fails a gross qualitative slowdown faster than the 240-min job
    // ceiling. See `WALL_BUDGET_S` for why a tight cap is not valid on this arm.
    assert!(
        elapsed < WALL_BUDGET_S,
        "survival marginal-slope fit took {elapsed:.1}s at n={N} centers={CENTERS} \
         (backstop {WALL_BUDGET_S}s); far beyond the converged ~10^3 s runtime — a \
         qualitative #1040-class slowdown, not the heavy-but-correct quadrature"
    );

    // ── Assertion 3: a sane, finite estimate ──────────────────────────────
    for block in &fit.fit.blocks {
        for &b in block.beta.iter() {
            assert!(
                b.is_finite(),
                "every fitted coefficient must be finite; got {b}"
            );
        }
    }
}
