//! Regression for #2670 / #2814 item 2: a Weibull proportional-hazards cohort
//! with HETEROGENEOUS delayed entry (`entry ~ U(0, 0.6·median)`, n = 1800)
//! fits through the real `gam fit` transformation path and is certified —
//! for both a parametric and a smooth covariate effect.
//!
//! # What refused it
//!
//! Under delayed entry the per-row Hessian is
//! `exp(η_exit)a₁a₁ᵀ − exp(η_entry)a₀a₀ᵀ + δ·ddᵀ/s²`: indefinite away from the
//! mode. The symmetric solve behind every inner Newton direction fell back
//! LLᵀ → LDLᵀ → LBLᵀ and so returned the exact SADDLE step on it; the inner
//! solves at the selector's seed then stalled and the outer selection refused.
//! With the direction taken on the block's descent curvature the selection
//! certifies, and the second refusal appeared: the fixed-λ solve that mints
//! the fit re-derived the mode from the cold structural seed (the corner of
//! the `γ ≥ 0` box), crawled, and left on a 20-step objective plateau with
//! `‖Pg‖ = 2.3e2` — although the selector had just certified β̂(ρ̂). The final
//! solve now starts from that certified mode and re-certifies it.
//!
//! The `~ x` arm and the `~ s(x)` arm are both asserted: they exercise the
//! parametric and the penalized covariate block through the same delayed-entry
//! baseline, and both were refused before the fix.

use std::path::Path;
use std::process::Command;

const N: usize = 1_800;

fn build_dataset(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1.0e-12, 1.0 - 1.0e-12)
    };
    // Weibull PH: Λ(t | x) = (t/σ)^k · exp(β x), k = 1.5, σ = 4, β = 0.8.
    let (k, sigma, beta) = (1.5_f64, 4.0_f64, 0.8_f64);
    let median = sigma * (std::f64::consts::LN_2).powf(1.0 / k);
    let mut entry = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    let mut x = Vec::with_capacity(N);
    while x.len() < N {
        let xi = -1.0 + 2.0 * next_u01();
        let u = next_u01();
        let t = sigma * (-u.ln() / (beta * xi).exp()).powf(1.0 / k);
        let e = 0.6 * median * next_u01();
        if t <= e {
            // Left truncation: a subject who fails before entry is never observed.
            continue;
        }
        let c = e + (-next_u01().ln()) * 6.0;
        entry.push(e);
        exit.push(t.min(c));
        event.push(if t <= c { 1.0 } else { 0.0 });
        x.push(xi);
    }
    (entry, exit, event, x)
}

fn write_training_csv(path: &Path, entry: &[f64], exit: &[f64], event: &[f64], x: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                format!("{:.12}", entry[i]),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn left_truncated_weibull_ph_heterogeneous_entry_fits_and_certifies_2670() {
    let (entry, exit, event, x) = build_dataset(2814);
    let min_entry = entry.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_entry = entry.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_entry > 0.0 && max_entry > 10.0 * min_entry,
        "the fixture must carry genuinely heterogeneous delayed entry, got [{min_entry}, {max_entry}]"
    );
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    write_training_csv(&train_path, &entry, &exit, &event, &x);
    for (idx, formula) in ["Surv(entry, exit, event) ~ x", "Surv(entry, exit, event) ~ s(x)"]
        .iter()
        .enumerate()
    {
        let model_path = dir.path().join(format!("model_{idx}.json"));
        let out = Command::new(gam::gam_binary!())
            .arg("fit")
            .arg(&train_path)
            .arg(formula)
            .arg("--out")
            .arg(&model_path)
            .output()
            .expect("spawn gam fit");
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(12).collect();
        assert!(
            out.status.success(),
            "left-truncated Weibull-PH fit `{formula}` was refused (status {:?}):\n{}",
            out.status.code(),
            tail.into_iter().rev().collect::<Vec<_>>().join("\n")
        );
        assert!(
            model_path.is_file(),
            "`{formula}` reported success without writing {}",
            model_path.display()
        );
    }
}
