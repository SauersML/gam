//! gam#979 regression: the survival marginal-slope fit MUST return (or raise)
//! catchably in bounded wall-clock time, never hang, even when its constrained
//! joint-Newton cannot certify convergence (the monotonicity-pinned baseline
//! whose active-set QP never certifies, so seed screening escalates to an
//! uncapped cycle budget while every seed rejects). Before the fix this ran to a
//! hard external timeout (issue #979: rc=124 at ~2000s, uncatchable). The fix
//! arms a configurable wall-clock deadline (here via FitConfig) checked at the
//! joint-Newton cycle-loop chokepoint, so the public API returns in bounded time.
//!
//! The guarantee under test is BOUNDED RETURN, not convergence: a usable fit OR
//! a catchable error both pass; only a hang fails.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const N_PCS: usize = 3;

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}
#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn build_dataset(n: usize) -> gam::inference::data::EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }
    headers.push("sex".to_string());
    let mut st: u64 = 0xD0E1_2345_6789_ABCD;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let pcs: Vec<f64> = (0..N_PCS).map(|_| next_gauss(&mut st) * 0.5).collect();
        let prs = next_gauss(&mut st);
        let sex = if next_unit(&mut st) < 0.5 { 1.0 } else { 0.0 };
        let entry = 40.0 + 5.0 * next_unit(&mut st);
        let exit = entry + 0.5 + 8.0 * next_unit(&mut st);
        let score = 0.3 * prs + 0.4 * pcs[0] - 0.3 * pcs.get(1).copied().unwrap_or(0.0)
            + 0.2 * pcs.get(2).copied().unwrap_or(0.0)
            + 0.15 * sex
            + 0.2 * next_gauss(&mut st);
        let event = if score > 0.0 { 1 } else { 0 };
        let mut rec = vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
        ];
        for p in &pcs {
            rec.push(p.to_string());
        }
        rec.push(sex.to_string());
        rows.push(StringRecord::from(rec));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode survival margslope dataset")
}

#[test]
fn survival_marginal_slope_returns_bounded_not_hang_979() {
    init_parallelism();
    let n = 1200usize;
    let centers = 12usize;
    let budget_secs = 20.0_f64;
    let data = build_dataset(n);
    let pcs: Vec<String> = (0..N_PCS).map(|i| format!("PC{}", i + 1)).collect();
    let duchon = format!("duchon({}, centers={}, order=1)", pcs.join(", "), centers);
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + sex", duchon);
    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(duchon),
        baseline_target: "linear".to_string(),
        gpu_policy: gam::gpu::GpuPolicy::Off,
        outer_wall_clock_budget_secs: Some(budget_secs),
        ..FitConfig::default()
    };

    // Run the fit on a worker thread; the test thread enforces the timeout so a
    // regression (a true hang) FAILS the test rather than hanging the suite.
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let start = Instant::now();
        let outcome = fit_from_formula(&formula, &data, &config);
        tx.send((outcome.is_ok(), start.elapsed().as_secs_f64()))
            .ok();
    });

    // Generous allowance over the budget for post-deadline cleanup + final refit.
    let hard_limit = Duration::from_secs_f64(budget_secs + 120.0);
    match rx.recv_timeout(hard_limit) {
        Ok((ok, secs)) => {
            eprintln!("[979-bounded] survival fit returned in {secs:.1}s (ok={ok})");
            assert!(
                secs < budget_secs + 110.0,
                "survival marginal-slope fit took {secs:.1}s with a {budget_secs:.0}s budget \
                 — bounded-return regression (#979)"
            );
        }
        Err(_) => panic!(
            "survival marginal-slope fit did NOT return within {}s — #979 bounded-return \
             regression (the fit hung)",
            budget_secs + 120.0
        ),
    }
}
