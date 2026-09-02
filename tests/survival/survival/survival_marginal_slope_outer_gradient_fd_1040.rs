//! Discriminating diagnostic for issue #1040: does the survival marginal-slope
//! outer REML/LAML loop fail to converge because of an objective↔gradient
//! DESYNC (a bug — analytic gradient disagrees with a finite-difference of the
//! criterion, so the trust region chases a phantom descent direction forever),
//! or because of weak IDENTIFIABILITY (a genuinely flat valley — analytic ≈ FD
//! everywhere but a near-zero outer-Hessian eigenvalue)?
//!
//! The fork is a finite difference of the outer gradient at a fixed θ,
//! component by component. The generic outer runner exposes an opt-in
//! structured record containing its real bounded seed, exact ρ/ψ layout,
//! analytic gradient, finite-difference gradient, and stencil steps. This test
//! drives a small survival-MS fit and consumes that typed evidence directly.
//!
//! The audit runs at θ₀ before the potentially long joint spatial loop. Its own
//! `max_outer_iter` is capped, while prerequisite rho-only profiles retain the
//! production budget required to reach that joint problem.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

const N: usize = 800;

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    init_parallelism();
}

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn next_gamma_alpha_ge_one(state: &mut u64, alpha: f64, scale: f64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = next_gauss(state);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = next_unit(state).max(1.0e-12);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return scale * d * v;
        }
    }
}
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "sex".to_string(),
        "prs_z".to_string(),
        "PC1".to_string(),
        "PC2".to_string(),
    ];
    let mut state = 0xA0B10B_u64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    let mut pc1 = Vec::with_capacity(N);
    let mut pc2 = Vec::with_capacity(N);
    let mut sex = Vec::with_capacity(N);
    let mut prs_z = Vec::with_capacity(N);
    let mut entry_age = Vec::with_capacity(N);
    let mut exit_age = Vec::with_capacity(N);

    for _ in 0..N {
        pc1.push(clip(0.024 + 0.042 * next_gauss(&mut state), -0.34, 0.12));
        pc2.push(clip(0.081 + 0.030 * next_gauss(&mut state), -0.19, 0.15));
        sex.push(if next_unit(&mut state) < 0.39 {
            1.0
        } else {
            0.0
        });
        prs_z.push(next_gauss(&mut state));
        let entry = clip(45.08 + 18.0 * next_gauss(&mut state), 1.55, 121.96);
        let followup = next_gamma_alpha_ge_one(&mut state, 1.7, 4.4) + 0.05;
        entry_age.push(entry);
        exit_age.push((entry + followup).min(122.47).max(entry + 1.0e-3));
    }

    let prs_mean = prs_z.iter().copied().sum::<f64>() / N as f64;
    let prs_var = prs_z.iter().map(|x| (x - prs_mean).powi(2)).sum::<f64>() / N as f64;
    let prs_sd = prs_var.sqrt().max(1.0e-9);
    for z in &mut prs_z {
        *z = (*z - prs_mean) / prs_sd;
    }
    let pc1_mean = pc1.iter().copied().sum::<f64>() / N as f64;
    let pc2_mean = pc2.iter().copied().sum::<f64>() / N as f64;

    let mut event_score = Vec::with_capacity(N);
    for i in 0..N {
        event_score.push(
            0.34 * prs_z[i] + 0.15 * sex[i] + 2.4 * (pc1[i] - pc1_mean) - 1.6 * (pc2[i] - pc2_mean)
                + next_gauss(&mut state),
        );
    }
    let mut sorted = event_score.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[N / 2];

    for i in 0..N {
        let record: Vec<String> = vec![
            entry_age[i].to_string(),
            exit_age[i].to_string(),
            if event_score[i] >= median { "1" } else { "0" }.to_string(),
            sex[i].to_string(),
            prs_z[i].to_string(),
            pc1[i].to_string(),
            pc2[i].to_string(),
        ];
        rows.push(StringRecord::from(record));
    }
    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode small survival marginal-slope dataset")
}

#[test]
fn survival_marginal_slope_outer_gradient_fd_audit_matern() {
    run_basis("matern(PC1, PC2, centers=4)");
}

#[test]
fn survival_marginal_slope_outer_gradient_fd_audit_duchon() {
    run_basis("duchon(PC1, PC2, centers=4, order=1, length_scale=1.0)");
}
