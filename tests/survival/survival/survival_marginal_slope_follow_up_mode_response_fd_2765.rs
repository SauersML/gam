//! gam#2765 / gam#2767 end-to-end gate: on a follow-up-varying slope the
//! criterion's coefficient mode response `dβ̂/dψ` must match a finite difference
//! of the fit's own β̂.
//!
//! The unit gates in `psi_terms_fd_tests` difference `D_β H[δ]` and
//! `D²_β H[u,v]` against the family's own joint Hessian, which is where the
//! defect was: `add_pullback_primary_hessian` pulled the row Hessian back
//! through ONE slope channel, so on a varying slope every consumer of that
//! pullback differentiated a different model. This gate closes the same defect
//! from the other end, through a real fit, because `D_β H` is what builds the
//! Jeffreys curvature `H_Φ` and its second-order completion — hence the operator
//! the mode response is solved against. A wrong pullback therefore shows up as a
//! wrong `dβ̂/dψ`, and that is a quantity the outer runner already publishes
//! beside its own Ridders-certified finite difference.
//!
//! Measured at the acceptance fixture's shape (`n = 400`, Weibull baseline,
//! `slope_time_k = 4`): `3.3e-2` and `3.2e-2` relative before the repair,
//! `5.0e-8` and `8.6e-9` after — six orders, on a quantity whose oracle is the
//! same inner solve the fit runs. The `1e-5` bar below sits four orders above
//! the repaired value and three below the broken one, so it cannot be cleared by
//! a partial fix.
//!
//! This grades the mode response, not the total outer gradient. They are
//! separate contracts: this fixture isolates the coefficient response that the
//! follow-up margin changes, while the complete profiled-gradient calculus is
//! covered by its own outer-gradient gates.

//!
//! ─── Restored 2026-09-04 (#2818) ───
//!
//! This file kept its module doc and lost its test: `c0a21b554` deleted the
//! body because it no longer compiled, and it no longer compiled because
//! `d484a091a` had deleted `gam::estimate::enable_outer_gradient_fd_capture` —
//! the ψ-only arming wrapper — under a criterion ("no symbol for it in the CLI
//! or pyffi binary") that a Rust-library API called only from tests and examples
//! satisfies by construction. `tests/survival/survival/mod.rs` still declared
//! this module the whole time, so the census stayed green over an empty file.
//!
//! The rebuild arms through `enable_outer_gradient_fd_capture_over_theta`, the
//! surviving public entry point, which is what the kept
//! `examples/probe_2765_outer_gradient_fd.rs` already uses. It grades the ρ
//! block as well as ψ; this gate reads only the ψ-block mode-response atoms, so
//! the wider arming is a superset of what was asked for before and changes no
//! assertion. Every other name here is a production entry point.

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

/// Small enough to keep this gate a few minutes, large enough that the outer
/// runner reaches a bounded joint seed with both ψ coordinates enrolled.
const N: usize = 200;
const SLOPE_TIME_DEGREE: usize = 2;
const SLOPE_TIME_K: usize = 4;
const SLOPE_LEVEL: f64 = 0.85;
const SLOPE_TREND: f64 = -0.32;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;

#[test]
fn survival_marginal_slope_follow_up_mode_response_matches_fd_2765() {
    // The fixture builders live inside the test as closures rather than as
    // module-level `fn`s: a symbol-table reachability sweep is vacuously true of
    // any test-only item, and that is what orphaned this gate the first time.
    let next_unit = |state: &mut u64| -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    };
    let next_gauss = |state: &mut u64| -> f64 {
        let u1 = next_unit(state).max(1e-12);
        let u2 = next_unit(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let planted_eta = |time: f64, z: f64| -> f64 {
        let slope = SLOPE_LEVEL + SLOPE_TREND * time.ln();
        let location = LOCATION_LEVEL + LOCATION_TREND * time.ln();
        location * (1.0 + slope * slope).sqrt() + slope * z
    };
    let normal_quantile = |p: f64| -> f64 {
        let (mut low, mut high) = (-12.0_f64, 12.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if gam_math::probability::normal_cdf(mid) < p {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    };
    let planted_event_time = |u: f64, z: f64| -> f64 {
        let target = -normal_quantile(u);
        let (mut low, mut high) = (-6.0_f64, 6.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if planted_eta(mid.exp(), z) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        (0.5 * (low + high)).exp()
    };

    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    gam::estimate::enable_outer_gradient_fd_capture_over_theta(1);

    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2765_2767_5CA1_AB1E_u64;
    let mut raw_scores: Vec<f64> = Vec::with_capacity(N);
    let mut draws: Vec<f64> = Vec::with_capacity(N);
    let mut censor: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        raw_scores.push(next_gauss(&mut state));
        draws.push(next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6));
        censor.push(next_unit(&mut state));
    }
    let mean = raw_scores.iter().sum::<f64>() / N as f64;
    let variance = raw_scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / N as f64;
    let sd = variance.sqrt().max(1e-12);
    let scores: Vec<f64> = raw_scores.iter().map(|v| (v - mean) / sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for index in 0..N {
        let z = scores[index];
        let event_time = planted_event_time(draws[index], z);
        let censor_time = 0.35 + 5.0 * censor[index];
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the #2765 fixture");

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        slope_time_k: Some(SLOPE_TIME_K),
        slope_time_degree: SLOPE_TIME_DEGREE,
        time_num_internal_knots: 3,
        baseline_target: "weibull".to_string(),
        // The audit fires at the first bounded joint seed; capping the joint
        // problem after it keeps this a gate rather than a benchmark.
        spatial_optimization: gam::smooth::SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 2,
            ..gam::smooth::SpatialLengthScaleOptimizationOptions::default()
        },
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    match fit_from_formula("Surv(time, event) ~ 1", &data, &config) {
        Ok(_) => eprintln!("[2765-MODE] fit returned Ok"),
        Err(error) => eprintln!("[2765-MODE] fit returned Err (the audit still ran): {error}"),
    }

    let audit = gam::estimate::take_outer_gradient_fd_capture()
        .expect("the outer runner must publish structured analytic-vs-FD evidence");
    assert!(
        audit.psi_dim >= 1,
        "a Weibull baseline enrolls its chart coordinates as ψ; got psi_dim={}",
        audit.psi_dim,
    );
    let atoms = audit.decomposition.atoms().unwrap_or_else(|| {
        panic!(
            "the survival marginal-slope criterion IS a REML assembly, so its audit must \
             carry the atom breakdown; got: {:?}",
            audit.decomposition
        )
    });

    for j in 0..audit.psi_dim {
        eprintln!(
            "[2765-MODE] psi_i={j} mode response: analytic_norm={:.6e} fd_norm={:.6e} \
             rel={:.3e} max_abs={:.3e}",
            atoms.analytic_mode_response_norm[j],
            atoms.finite_difference_mode_response_norm[j],
            atoms.mode_response_relative_error[j],
            atoms.mode_response_max_abs_error[j],
        );
    }
    for j in 0..audit.psi_dim {
        // A zero response is not evidence: it would agree with anything.
        assert!(
            atoms.analytic_mode_response_norm[j] > 1e-6,
            "psi coordinate {j} publishes a numerically zero mode response \
             ({:.3e}), so this gate would pass vacuously",
            atoms.analytic_mode_response_norm[j],
        );
        assert!(
            atoms.mode_response_relative_error[j] < 1e-5,
            "psi coordinate {j}: the coefficient mode response dbeta/dpsi disagrees with \
             its own finite difference by {:.3e} relative (max_abs={:.3e}, \
             analytic_norm={:.6e}, fd_norm={:.6e}) — on a follow-up-varying slope this is \
             the signature of a pullback that reads the slope through one channel \
             (gam#2765)",
            atoms.mode_response_relative_error[j],
            atoms.mode_response_max_abs_error[j],
            atoms.analytic_mode_response_norm[j],
            atoms.finite_difference_mode_response_norm[j],
        );
    }
}
