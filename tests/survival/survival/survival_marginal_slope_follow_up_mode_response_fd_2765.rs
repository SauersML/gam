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
//! This grades the mode response, NOT the total outer gradient: that total still
//! carries the `logdet_h` disagreement the `#979`/`#1040` lane owns, which
//! reproduces identically with `slope_time_k` unset and is therefore not this
//! issue's to assert on.

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

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn planted_eta(time: f64, z: f64) -> f64 {
    let slope = SLOPE_LEVEL + SLOPE_TREND * time.ln();
    let location = LOCATION_LEVEL + LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| gam_math::probability::normal_cdf(x);
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

fn build_dataset() -> gam::inference::data::EncodedDataset {
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
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2765 fixture")
}

