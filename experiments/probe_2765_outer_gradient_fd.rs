//! gam#2765 / gam#2767 probe: is the outer criterion's ψ gradient right at the
//! acceptance fixture's OWN model shape?
//!
//! `28b41a025` left the fixture running (0 → 52 outer iterations) and refusing
//! with `|Pg| = 4.550e0`, `railed = [2]`, `hessian_psd = NO`. Two of the five
//! θ coordinates are the Weibull baseline chart's own ψ, and an earlier probe
//! on this thread measured the failing search direction to be essentially pure
//! ψ (ρ moved `0.007` across an entire backtracking). The ρ coordinates that
//! moved are the two smoothing asymptotes (`ρ₀ ≈ +9.3`, `ρ₁ ≈ −10.0`), whose
//! exponential tail law puts their gradients at `e^{∓ρ} ≈ 10⁻⁴` — four orders
//! below the residual that refuses. So the residual is carried by ψ, and the
//! question is whether the analytic ψ gradient of THIS criterion is right.
//!
//! `tests/survival/survival/survival_marginal_slope_outer_gradient_fd_1040.rs`
//! asks exactly that question, but on a different model shape: a *design* ψ
//! (matérn / duchon length scale) with a LINEAR baseline. The fixture's ψ is a
//! *baseline-chart* ψ, which reaches η through the three offset channels of the
//! location index and touches no design at all. Those are two structurally
//! different lowerings, and the 1040 audit says nothing about this one.
//!
//! So: the same self-certifying Ridders audit, armed at the fixture's shape.
//! Run with `RUST_LOG=info` and read `[2765-FD]`.

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

const SLOPE_TIME_DEGREE: usize = 2;
const SLOPE_TIME_K: usize = 4;
const SLOPE_LEVEL: f64 = 0.85;
const SLOPE_TREND: f64 = -0.32;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn planted_slope(time: f64) -> f64 {
    SLOPE_LEVEL + SLOPE_TREND * time.ln()
}

fn planted_eta(time: f64, z: f64) -> f64 {
    let slope = planted_slope(time);
    let location = LOCATION_LEVEL + LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn planted_event_time(u: f64, z: f64) -> f64 {
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
}

fn build_dataset(n: usize) -> gam::inference::data::EncodedDataset {
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2765_2767_5CA1_AB1E_u64;

    let mut raw_scores: Vec<f64> = Vec::with_capacity(n);
    let mut draws: Vec<f64> = Vec::with_capacity(n);
    let mut censor: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        raw_scores.push(next_gauss(&mut state));
        draws.push(next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6));
        censor.push(next_unit(&mut state));
    }
    let mean = raw_scores.iter().sum::<f64>() / n as f64;
    let variance = raw_scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = variance.sqrt().max(1e-12);
    let scores: Vec<f64> = raw_scores.iter().map(|v| (v - mean) / sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for index in 0..n {
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
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2765 fixture")
}

fn main() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);
    // `static` drops the follow-up margin, which is the control that says
    // whether a ψ defect here belongs to the follow-up frame or predates it.
    let arm = std::env::args().nth(2).unwrap_or_else(|| "dynamic".to_string());
    let time_k = if arm == "static" {
        None
    } else {
        Some(SLOPE_TIME_K)
    };

    // Grade the WHOLE θ vector, not just ψ. The ρ block and the ψ block share
    // the criterion's moving-Hessian machinery (the trace kernel and the
    // mode-response drift `D_β H[v]`) and differ only in their frozen drift, so
    // a `logdet_h` disagreement present in one and absent in the other says
    // which half of `frozen + mode_response` is wrong — the question a ψ-only
    // audit cannot ask.
    gam::estimate::enable_outer_gradient_fd_capture_over_theta(1);
    let data = build_dataset(n);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        slope_time_k: time_k,
        slope_time_degree: SLOPE_TIME_DEGREE,
        time_num_internal_knots: 3,
        baseline_target: "weibull".to_string(),
        // The audit runs at the first bounded seed; capping the joint problem
        // after it keeps this probe minutes rather than half an hour.
        spatial_optimization: gam::smooth::SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 2,
            ..gam::smooth::SpatialLengthScaleOptimizationOptions::default()
        },
        ..FitConfig::default()
    };

    eprintln!("[2765-FD] n={n} arm={arm} slope_time_k={time_k:?}");
    let started = std::time::Instant::now();
    match fit_from_formula("Surv(time, event) ~ 1", &data, &config) {
        Ok(_) => eprintln!(
            "[2765-FD] fit returned Ok in {:.1}s",
            started.elapsed().as_secs_f64()
        ),
        Err(error) => eprintln!(
            "[2765-FD] fit returned Err after {:.1}s: {error}",
            started.elapsed().as_secs_f64()
        ),
    }

    let Some(audit) = gam::estimate::take_outer_gradient_fd_capture() else {
        eprintln!("[2765-FD] NO AUDIT RECORD — the capture never armed at a bounded seed");
        return;
    };

    eprintln!(
        "[2765-FD] theta={:?} rho_dim={} psi_dim={} cost={:.9e}",
        audit.theta.to_vec(),
        audit.rho_dim,
        audit.psi_dim,
        audit.cost,
    );
    for j in 0..audit.psi_dim {
        let analytic = audit.analytic_psi_gradient[j];
        let fd = audit.finite_difference_psi_gradient[j];
        let gap = (analytic - fd).abs();
        let scale = analytic.abs().max(fd.abs()).max(1e-6);
        eprintln!(
            "[2765-FD] psi_i={j} analytic={analytic:+.6e} fd={fd:+.6e} gap={gap:.3e} \
             rel={:.3e} step={:.3e} oracle_unc={:.3e} order={}",
            gap / scale,
            audit.psi_steps[j],
            audit.psi_fd_uncertainty[j],
            audit.psi_fd_orders[j],
        );
    }
    match audit.rho.as_ref() {
        None => eprintln!("[2765-FD] no rho block in the record"),
        Some(rho) => {
            for j in 0..audit.rho_dim {
                let analytic = rho.analytic_gradient[j];
                let fd = rho.finite_difference_gradient[j];
                let gap = (analytic - fd).abs();
                let scale = analytic.abs().max(fd.abs()).max(1e-6);
                eprintln!(
                    "[2765-FD] rho_i={j} analytic={analytic:+.6e} fd={fd:+.6e} gap={gap:.3e} \
                     rel={:.3e} step={:.3e} oracle_unc={:.3e} order={}",
                    gap / scale,
                    rho.steps[j],
                    rho.fd_uncertainty[j],
                    rho.fd_orders[j],
                );
                eprintln!(
                    "[2765-FD] rho_i={j} analytic atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                     (frozen={:+.6e} mode_response={:+.6e}) \
                     logdet_s={:+.6e} kkt={:+.6e} audit_total={:+.6e}",
                    rho.analytic_fixed_beta[j],
                    rho.analytic_logdet_h[j],
                    rho.analytic_frozen_logdet_h[j],
                    rho.analytic_mode_response_logdet_h[j],
                    rho.analytic_logdet_s[j],
                    rho.analytic_kkt[j],
                    rho.analytic_audit_total[j],
                );
                eprintln!(
                    "[2765-FD] rho_i={j} scalar-FD atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                     logdet_s={:+.6e} kkt={:+.6e}",
                    rho.finite_difference_fixed_beta[j],
                    rho.finite_difference_logdet_h[j],
                    rho.finite_difference_logdet_s[j],
                    rho.finite_difference_kkt[j],
                );
            }
        }
    }
    match audit.curvature.as_ref() {
        None => eprintln!("[2765-FD] no curvature audit (backend has no dense H)"),
        Some(curvature) => {
            eprintln!(
                "[2765-FD] curvature: criterion_half_logdet={:+.9e} dense_half_logdet={:+.9e} \
                 gap={:.3e} tangent_dim={:?}",
                curvature.criterion_half_logdet,
                curvature.dense_half_logdet,
                (curvature.criterion_half_logdet - curvature.dense_half_logdet).abs(),
                curvature.tangent_dim,
            );
            eprintln!(
                "[2765-FD] face curvature ZtHZ = {:?}",
                curvature.face_curvature
            );
            for j in 0..curvature.drift_max_abs_error.len() {
                let label = if j < audit.rho_dim {
                    format!("rho_{j}")
                } else {
                    format!("psi_{}", j - audit.rho_dim)
                };
                eprintln!(
                    "[2765-FD] drift {label}: max_abs_err={:.6e} rel={:.3e} \
                     |Hdot|max={:.6e} worst_entry={:?} face_move={:.3e} face_dims={:?}",
                    curvature.drift_max_abs_error[j],
                    curvature.drift_relative_error[j],
                    curvature.analytic_drift_max_abs[j],
                    curvature.drift_worst_entry[j],
                    curvature.face_drift_max_abs[j],
                    curvature.displaced_tangent_dim[j],
                );
                eprintln!(
                    "[2765-FD] drift {label} analytic ZtHdotZ = {:?}",
                    curvature.analytic_face_drift[j]
                );
                eprintln!(
                    "[2765-FD] drift {label} measured d(ZtHZ) = {:?}",
                    curvature.measured_face_drift[j]
                );
            }
        }
    }
    match audit.decomposition.atoms() {
        None => eprintln!("[2765-FD] no atom breakdown: {:?}", audit.decomposition),
        Some(atoms) => {
            for j in 0..audit.psi_dim {
                eprintln!(
                    "[2765-FD] psi_i={j} analytic atoms: fixed_beta={:+.6e} \
                     logdet_h={:+.6e} (frozen={:+.6e} mode_response={:+.6e}) \
                     logdet_s={:+.6e} kkt={:+.6e}",
                    atoms.fixed_beta_psi_gradient[j],
                    atoms.logdet_h_psi_gradient[j],
                    atoms.frozen_logdet_h_psi_gradient[j],
                    atoms.mode_response_logdet_h_psi_gradient[j],
                    atoms.logdet_s_psi_gradient[j],
                    atoms.kkt_psi_gradient[j],
                );
                eprintln!(
                    "[2765-FD] psi_i={j} scalar-FD atoms: fixed_beta={:+.6e} \
                     logdet_h={:+.6e} logdet_s={:+.6e} kkt={:+.6e}",
                    atoms.finite_difference_fixed_beta_psi_gradient[j],
                    atoms.finite_difference_logdet_h_psi_gradient[j],
                    atoms.finite_difference_logdet_s_psi_gradient[j],
                    atoms.finite_difference_kkt_psi_gradient[j],
                );
                eprintln!(
                    "[2765-FD] psi_i={j} mode response: analytic_norm={:.6e} \
                     fd_norm={:.6e} rel={:.3e} max_abs={:.3e}",
                    atoms.analytic_mode_response_norm[j],
                    atoms.finite_difference_mode_response_norm[j],
                    atoms.mode_response_relative_error[j],
                    atoms.mode_response_max_abs_error[j],
                );
            }
        }
    }
}
