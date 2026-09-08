//! gam#2765 control: does the criterion's `logdet_h` gradient atom disagree with
//! its own finite difference OUTSIDE the survival marginal-slope family?
//!
//! The #2765 measurement is that the whole outer-gradient gap is the `logdet_h`
//! atom, on BOTH the ψ block and the ρ block, while `fixed_beta` and `logdet_s`
//! are exact to six and seven digits. `logdet_h` is the only atom that reads the
//! coefficient mode response `dβ̂/dθ` — every other atom is an envelope
//! derivative at fixed β̂ — so the suspects are the shared moving-Hessian chain
//! (`½ tr(K · D_β H[dβ̂/dθ])`) and the pieces only the survival lane installs
//! (a live row-wise active set, and the Firth/Jeffreys curvature `H_Φ`).
//!
//! This control separates them. `y ~ matern(x1, x2)` under a BINOMIAL likelihood
//! is `c`-nontrivial — the IRLS weights depend on η, so `D_β H ≠ 0` and the
//! mode-response term is live — but it carries no Jeffreys term, no active
//! constraints, and no custom family: it is the shipped GLM REML assembly. If
//! its `logdet_h` atoms match their finite differences, the shared chain is
//! sound and the defect belongs to the survival lane; if they do not, the
//! defect is in machinery every penalized non-Gaussian fit in the crate uses.
//!
//! The Gaussian sibling (`matern_2d_iso_kappa_outer_gradient_matches_fd`) cannot
//! answer this: under the identity link `c ≡ 0`, `D_β H ≡ 0`, and the
//! mode-response term is identically zero.

use gam::utils::splitmix64;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn truth(a: f64, b: f64) -> f64 {
    1.6 * (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize) -> gam::inference::data::EncodedDataset {
    let mut state = 0x9A7E_7212_2765_u64;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let a = next_unit(&mut state);
        let b = next_unit(&mut state);
        let eta = truth(a, b);
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = if next_unit(&mut state) < p { 1u8 } else { 0u8 };
        records.push(csv::StringRecord::from(vec![
            y.to_string(),
            format!("{a:.9}"),
            format!("{b:.9}"),
        ]));
    }
    encode_recordswith_inferred_schema(
        vec!["y".to_string(), "x1".to_string(), "x2".to_string()],
        records,
    )
    .expect("encode the binomial matern control dataset")
}

fn main() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);
    let family = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "binomial".to_string());

    gam::estimate::enable_outer_gradient_fd_capture_over_theta(1);
    let data = build_dataset(n);
    let config = FitConfig {
        family: Some(family.clone()),
        spatial_optimization: gam::smooth::SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 2,
            ..gam::smooth::SpatialLengthScaleOptimizationOptions::default()
        },
        ..FitConfig::default()
    };

    eprintln!("[2765-CTRL] n={n} family={family} formula=\"y ~ matern(x1, x2)\"");
    match gam::fit_from_formula("y ~ matern(x1, x2)", &data, &config) {
        Ok(_) => eprintln!("[2765-CTRL] fit returned Ok"),
        Err(error) => eprintln!("[2765-CTRL] fit returned Err (audit still ran): {error}"),
    }

    let Some(audit) = gam::estimate::take_outer_gradient_fd_capture() else {
        eprintln!("[2765-CTRL] NO AUDIT RECORD");
        return;
    };
    eprintln!(
        "[2765-CTRL] theta={:?} rho_dim={} psi_dim={} cost={:.9e}",
        audit.theta.to_vec(),
        audit.rho_dim,
        audit.psi_dim,
        audit.cost,
    );
    for j in 0..audit.psi_dim {
        let analytic = audit.analytic_psi_gradient[j];
        let fd = audit.finite_difference_psi_gradient[j];
        let gap = (analytic - fd).abs();
        eprintln!(
            "[2765-CTRL] psi_i={j} analytic={analytic:+.6e} fd={fd:+.6e} gap={gap:.3e} \
             rel={:.3e} oracle_unc={:.3e} order={}",
            gap / analytic.abs().max(fd.abs()).max(1e-6),
            audit.psi_fd_uncertainty[j],
            audit.psi_fd_orders[j],
        );
    }
    if let Some(atoms) = audit.decomposition.atoms() {
        for j in 0..audit.psi_dim {
            eprintln!(
                "[2765-CTRL] psi_i={j} analytic atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                 (frozen={:+.6e} mode_response={:+.6e}) logdet_s={:+.6e} kkt={:+.6e}",
                atoms.fixed_beta_psi_gradient[j],
                atoms.logdet_h_psi_gradient[j],
                atoms.frozen_logdet_h_psi_gradient[j],
                atoms.mode_response_logdet_h_psi_gradient[j],
                atoms.logdet_s_psi_gradient[j],
                atoms.kkt_psi_gradient[j],
            );
            eprintln!(
                "[2765-CTRL] psi_i={j} scalar-FD atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                 logdet_s={:+.6e} kkt={:+.6e}",
                atoms.finite_difference_fixed_beta_psi_gradient[j],
                atoms.finite_difference_logdet_h_psi_gradient[j],
                atoms.finite_difference_logdet_s_psi_gradient[j],
                atoms.finite_difference_kkt_psi_gradient[j],
            );
            eprintln!(
                "[2765-CTRL] psi_i={j} mode response: analytic_norm={:.6e} fd_norm={:.6e} \
                 rel={:.3e} max_abs={:.3e}",
                atoms.analytic_mode_response_norm[j],
                atoms.finite_difference_mode_response_norm[j],
                atoms.mode_response_relative_error[j],
                atoms.mode_response_max_abs_error[j],
            );
        }
    }
    match audit.rho.as_ref() {
        None => eprintln!("[2765-CTRL] no rho block in the record"),
        Some(rho) => {
            for j in 0..audit.rho_dim {
                let analytic = rho.analytic_gradient[j];
                let fd = rho.finite_difference_gradient[j];
                let gap = (analytic - fd).abs();
                eprintln!(
                    "[2765-CTRL] rho_i={j} analytic={analytic:+.6e} fd={fd:+.6e} gap={gap:.3e} \
                     rel={:.3e} oracle_unc={:.3e} order={}",
                    gap / analytic.abs().max(fd.abs()).max(1e-6),
                    rho.fd_uncertainty[j],
                    rho.fd_orders[j],
                );
                eprintln!(
                    "[2765-CTRL] rho_i={j} analytic atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                     (frozen={:+.6e} mode_response={:+.6e}) \
                     logdet_s={:+.6e} kkt={:+.6e}",
                    rho.analytic_fixed_beta[j],
                    rho.analytic_logdet_h[j],
                    rho.analytic_frozen_logdet_h[j],
                    rho.analytic_mode_response_logdet_h[j],
                    rho.analytic_logdet_s[j],
                    rho.analytic_kkt[j],
                );
                eprintln!(
                    "[2765-CTRL] rho_i={j} scalar-FD atoms: fixed_beta={:+.6e} logdet_h={:+.6e} \
                     logdet_s={:+.6e} kkt={:+.6e}",
                    rho.finite_difference_fixed_beta[j],
                    rho.finite_difference_logdet_h[j],
                    rho.finite_difference_logdet_s[j],
                    rho.finite_difference_kkt[j],
                );
            }
        }
    }
}
