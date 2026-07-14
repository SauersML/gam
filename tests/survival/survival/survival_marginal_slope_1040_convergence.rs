//! Exact survival counterpart to the binary #979 Matérn marginal/log-slope
//! regression. The public formula omits `length_scale` in both channels, so
//! this must enter the joint automatic-κ optimizer rather than the cheaper
//! fixed-geometry route.
//!
//! The fixture uses the issue-scale spatial shape (`centers=12`, `n=2500`, no
//! FLEX/time-wiggle family axes) and deterministic censored data. Its
//! contracts are semantic rather than tolerance patches:
//!   * a returned fit is the sealed SPEC-20 convergence certificate;
//!   * both returned term manifests must retain typed `Auto` ownership and a
//!     finite resolved scale, proving the automatic-κ route survived planning,
//!     optimization, and result freezing;
//!   * all fitted coefficients are finite.
//!
//! Runtime is printed as `[979-SURVIVAL]` telemetry. The invoking workflow owns
//! the wall-clock timeout, so a hang is killed externally instead of being
//! disguised by an in-process threshold that cannot interrupt a blocked fit.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam::terms::basis::{CenterStrategy, MaternLengthScale, MaternNu};
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec};
use std::time::Instant;

const N: usize = 2_500;
const CENTERS: usize = 12;

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

fn resolved_auto_matern_scale(spec: &TermCollectionSpec, channel: &str) -> f64 {
    assert_eq!(
        spec.smooth_terms.len(),
        1,
        "{channel} must contain exactly the formula's one Matérn term"
    );
    let SmoothBasisSpec::Matern { spec: matern, .. } = &spec.smooth_terms[0].basis else {
        panic!("{channel} resolved to a non-Matérn basis");
    };
    let CenterStrategy::FarthestPoint { num_centers } = &matern.center_strategy else {
        panic!(
            "{channel} must retain formula-selected farthest-point centers; got {:?}",
            matern.center_strategy
        );
    };
    assert_eq!(
        *num_centers, CENTERS,
        "{channel} must retain the requested center count"
    );
    assert!(
        matches!(matern.nu, MaternNu::FiveHalves),
        "{channel} must retain the formula default nu=5/2; got {:?}",
        matern.nu
    );
    let MaternLengthScale::Auto {
        resolved: Some(scale),
    } = matern.length_scale
    else {
        panic!(
            "{channel} must retain resolved Auto Matérn ownership after the fit; got {:?}",
            matern.length_scale
        );
    };
    assert!(
        scale.is_finite() && scale > 0.0,
        "{channel} automatic Matérn scale must be positive and finite; got {scale}"
    );
    scale
}

#[test]
fn survival_marginal_slope_auto_matern_logslope_centers12_converges() {
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

    let marginal_scale =
        resolved_auto_matern_scale(&fit.marginalspec_resolved, "marginal channel");
    let logslope_scale =
        resolved_auto_matern_scale(&fit.logslopespec_resolved, "log-slope channel");

    eprintln!(
        "[979-SURVIVAL] n={N} centers={CENTERS} total_s={elapsed:.3} \
         marginal_scale={marginal_scale:.6e} logslope_scale={logslope_scale:.6e} \
         outer_iters={} inner_cycles={} converged=certified auto_kappa=both \
         centers_strategy=farthest_point nu=5/2",
        fit.fit.outer_iterations, fit.fit.inner_cycles
    );

    // ── Assertion 1: the survival-MS outer loop converged ─────────────────
    // A returned fit is the sealed survival-MS convergence certificate.

    // ── Assertion 2: it terminated, not the unbounded hang ────────────────
    // Fit construction above is the convergence assertion: SPEC 20 forbids
    // minting a fit from a non-converged optimization. The runner's timeout is
    // the hang assertion.

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
