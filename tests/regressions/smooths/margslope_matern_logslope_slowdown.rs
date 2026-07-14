//! Exact public-API regression for #979: a Bernoulli marginal-slope/probit
//! model with the same omitted-scale Matérn surface in the marginal and
//! log-slope channels.
//!
//! The old fixture manually assembled a different estimator: equal-mass
//! centers, ν=3/2, and direct fit-request internals. These cases deliberately
//! enter through the formula materializer used by the issue:
//!
//! ```text
//! event ~ matern(PC1, PC2, centers=...)
//! logslope = matern(PC1, PC2, centers=...)
//! family = bernoulli-marginal-slope, link = probit, z_column = z
//! ```
//!
//! A returned fit is the sealed SPEC-20 convergence certificate. Each case
//! also verifies that materialization and fit freezing preserve the exact
//! requested model: the selected farthest-point rows are frozen as an explicit
//! center matrix of the requested size, ν remains the default 5/2, and typed
//! `Auto` Matérn ownership retains a positive resolved scale in both channels.
//! Runtime is emitted as `[979-BINARY]` telemetry. The invoking workflow owns
//! the wall-clock timeout so a blocked fit is interrupted externally.

use csv::StringRecord;
use gam::terms::basis::{CenterStrategy, MaternLengthScale, MaternNu};
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::time::Instant;

const N: usize = 2_500;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736)
        * t
        + 0.254829592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = ["event", "z", "PC1", "PC2"]
        .iter()
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    let mut state = 0x9797_2026_0713_0001_u64;

    let mut pc1 = Vec::with_capacity(N);
    let mut pc2 = Vec::with_capacity(N);
    let mut z_raw = Vec::with_capacity(N);
    for _ in 0..N {
        pc1.push(4.0 * next_unit(&mut state) - 2.0);
        pc2.push(4.0 * next_unit(&mut state) - 2.0);
        z_raw.push(next_gauss(&mut state));
    }

    let z_mean = z_raw.iter().sum::<f64>() / N as f64;
    let z_variance = z_raw
        .iter()
        .map(|value| (value - z_mean).powi(2))
        .sum::<f64>()
        / N as f64;
    let z_sd = z_variance.sqrt().max(1e-12);

    let mut rows = Vec::with_capacity(N);
    for row in 0..N {
        let z = (z_raw[row] - z_mean) / z_sd;
        let marginal = (0.8 * pc1[row]).sin() + 0.5 * (0.6 * pc2[row]).cos();
        let log_slope = -0.55 + 0.16 * pc1[row] - 0.10 * pc2[row];
        let eta = marginal + log_slope.exp() * z;
        let probability = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        let event = u8::from(next_unit(&mut state) < probability);
        rows.push(StringRecord::from(vec![
            event.to_string(),
            z.to_string(),
            pc1[row].to_string(),
            pc2[row].to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows).expect("encode #979 binary dataset")
}

fn resolved_auto_matern_scale(
    spec: &TermCollectionSpec,
    channel: &str,
    expected_centers: usize,
) -> f64 {
    assert_eq!(
        spec.smooth_terms.len(),
        1,
        "{channel} must contain exactly the formula's one Matérn term"
    );
    let SmoothBasisSpec::Matern { spec: matern, .. } = &spec.smooth_terms[0].basis else {
        panic!("{channel} resolved to a non-Matérn basis");
    };
    let CenterStrategy::UserProvided(frozen_centers) = &matern.center_strategy else {
        panic!(
            "{channel} must own the explicit center matrix frozen by formula materialization; got {:?}",
            matern.center_strategy
        );
    };
    assert_eq!(
        frozen_centers.dim(),
        (expected_centers, 2),
        "{channel} frozen center matrix must retain the requested count and two-dimensional geometry"
    );
    assert!(
        frozen_centers.iter().all(|value| value.is_finite()),
        "{channel} frozen center matrix must be finite"
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

fn fit_issue_case(centers: usize) {
    init_parallelism();

    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();
    let matern = format!("matern(PC1, PC2, centers={centers})");
    let formula = format!("event ~ {matern}");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        z_column: Some("z".to_string()),
        logslope_formula: Some(matern),
        ..FitConfig::default()
    };

    let start = Instant::now();
    let result = fit_from_formula(&formula, &data, &config)
        .unwrap_or_else(|error| panic!("#979 binary centers={centers} fit failed: {error}"));
    let elapsed = start.elapsed().as_secs_f64();
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit result");
    };

    let marginal_scale = resolved_auto_matern_scale(
        &fit.marginalspec_resolved,
        "marginal channel",
        centers,
    );
    let logslope_scale = resolved_auto_matern_scale(
        &fit.logslopespec_resolved,
        "log-slope channel",
        centers,
    );
    for block in &fit.fit.blocks {
        for &coefficient in block.beta.iter() {
            assert!(
                coefficient.is_finite(),
                "every fitted coefficient must be finite; got {coefficient}"
            );
        }
    }

    eprintln!(
        "[979-BINARY] n={N} centers={centers} total_s={elapsed:.3} \
         marginal_scale={marginal_scale:.6e} logslope_scale={logslope_scale:.6e} \
         outer_iters={} inner_cycles={} converged=certified auto_kappa=both \
         requested_centers_strategy=farthest_point resolved_centers_strategy=user_provided nu=5/2",
        fit.fit.outer_iterations, fit.fit.inner_cycles
    );
}

#[test]
fn margslope_matern_logslope_timing() {
    fit_issue_case(4);
}

#[test]
fn margslope_matern_logslope_above_cliff() {
    fit_issue_case(12);
}

#[test]
fn margslope_matern_logslope_centers20_issue_scale() {
    fit_issue_case(20);
}
