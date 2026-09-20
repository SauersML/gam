//! gam#2768, gam#2926 — the marginal-slope fit must return the MARGINAL index,
//! end to end, on a score that is exactly `N(0, 1)` and conditionally shifted.
//!
//! # What is at stake
//!
//! The marginal-slope parameterisation `η = α(q, b) + b·z` earns its name from
//! the anchoring identity `E[Φ(α + b·z) | C] = Φ(q)`, which holds on the law of
//! `z | C` the fit anchors on. A score can be exactly standard normal overall
//! while every conditional law `z | C` is shifted, and the pooled adequacy check
//! cannot see the difference. This fixture builds exactly that:
//!
//! ```text
//!     x, ζ ~ N(0,1) independent,     z = m·x + √(1−m²)·ζ
//! ```
//!
//! `z ~ N(0,1)` exactly; `E[z | x] = m·x`. The outcome is generated from the
//! marginal-slope model **on ζ**, so the truth is `q = β₀ + β_x·x` with a
//! constant slope `b` on ζ — and `P(Y=1 | x) = Φ(β₀ + β_x·x)` exactly.
//!
//! On the raw `z` axis the same outcome model is
//!
//! ```text
//!     η = q·c(b) + b·ζ = [q·c(b) − b′·m·x] + b′·z,     b′ = b/√(1−m²),
//! ```
//!
//! and with `z | x ~ N(m·x, 1−m²)` its anchor is exactly `q`. So a fit that
//! anchors on the CONDITIONAL law of `z` returns the marginal index with slope
//! `b′`; a fit that anchors the raw axis on `N(0, 1)` gets the `b(C)·m(C)`
//! leakage in the influence channel instead — marginal x-coefficient
//! `β_x·c(b)/c(b′) − (b·m/√(1−m²))/c(b′) = 0.107` against a truth of `0.500`.
//!
//! # The arms
//!
//! * the default on `z`: the conditional law moves on the span, so the moving-law
//!   certificate chooses among the Gaussian, location-scale and local laws; on this
//!   location-scale fixture it keeps the location-scale Gaussian law, whose slope
//!   lives on the `ζ` axis;
//! * the default on `ζ`: the law does not move, so one global law;
//! * `latent_measure = "conditional-location-scale"` on `z`: the declared
//!   location-scale law, whose slope lives on the `ζ` axis;
//! * `latent_measure = "gaussian"` on `z`: refused, because `E[z | x]` moves.
//!
//! # Why this fixture is Bernoulli
//!
//! The gate is one shared object serving both marginal-slope families, so an
//! end-to-end estimand gate on either one gates the shared arithmetic. The
//! Bernoulli family is the one whose outer search is well conditioned at this
//! size; the survival family's own wiring is gated at its seams.

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const N: usize = 6_000;
/// `Corr(z, x)`, and the conditional-mean slope the gate has to find.
const M_SHIFT: f64 = 0.6;
/// True conditional slope on the standardised latent score.
const TRUE_SLOPE: f64 = 0.6;
/// True marginal-index coefficients.
const TRUE_BETA_X: f64 = 0.5;
const TRUE_INTERCEPT: f64 = -0.2;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn standardized(mut v: Vec<f64>) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    for value in v.iter_mut() {
        *value = (*value - mean) / sd;
    }
    v
}

struct Fixture {
    dataset: gam::inference::data::EncodedDataset,
    x: Vec<f64>,
}

fn build_fixture() -> Fixture {
    let headers = ["y", "z", "zeta", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2768_BEEF_5EED_0001;
    let x = standardized((0..N).map(|_| next_gauss(&mut state)).collect());
    let zeta = standardized((0..N).map(|_| next_gauss(&mut state)).collect());
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut positives = 0usize;
    for row in 0..N {
        let z = M_SHIFT * x[row] + residual_sd * zeta[row];
        let q = TRUE_INTERCEPT + TRUE_BETA_X * x[row];
        let eta = q * c_true + TRUE_SLOPE * zeta[row];
        let y = u8::from(next_unit(&mut state) < gam::probability::normal_cdf(eta));
        positives += usize::from(y == 1);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z.to_string(),
            zeta[row].to_string(),
            x[row].to_string(),
        ]));
    }
    eprintln!(
        "[2768] n={N} positives={positives} ({:.1}%)",
        100.0 * positives as f64 / N as f64
    );
    Fixture {
        dataset: encode_recordswith_inferred_schema(headers, rows).expect("encode #2768 fixture"),
        x,
    }
}

struct Arm {
    /// `∂(X_m β_m)/∂x`, read by projection rather than from a coefficient index:
    /// the marginal block carries whatever identifiability chart the frozen
    /// joint build chose, so the index is not a stable contract and the
    /// projection is.
    marginal_x_slope: f64,
    mean_slope: f64,
    law: &'static str,
    calibrated: bool,
    /// `D̂` of the closed-form certificate, when the adequacy screen passed.
    /// The closed-form certificate's `(D̂, null tail)`, when one was taken.
    certificate: Option<(f64, Option<f64>)>,
}

fn slope_on(values: &[f64], x: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let v_mean = values.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..values.len() {
        let dx = x[i] - x_mean;
        cov += dx * (values[i] - v_mean);
        var += dx * dx;
    }
    cov / var
}

fn config(z_column: &str, latent_measure: Option<&str>) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some(z_column.to_string()),
        slope_formula: Some("1".to_string()),
        latent_measure: latent_measure.map(str::to_string),
        ..FitConfig::default()
    }
}

fn fit_arm(fixture: &Fixture, z_column: &str, latent_measure: Option<&str>) -> Arm {
    let result = fit_from_formula("y ~ x", &fixture.dataset, &config(z_column, latent_measure))
        .unwrap_or_else(|e| {
            panic!("bernoulli marginal-slope fit on z_column={z_column} law={latent_measure:?}: {e}")
        });
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit for z_column={z_column}");
    };
    let marginal_eta = fit.marginal_design.design.dot(&fit.fit.blocks[0].beta);
    let slope_eta = fit.slope_design.design.dot(&fit.fit.blocks[1].beta);
    let mean_slope =
        fit.baseline_slope + slope_eta.iter().sum::<f64>() / slope_eta.len() as f64;
    Arm {
        marginal_x_slope: slope_on(
            marginal_eta
                .as_slice()
                .expect("marginal eta is standard layout"),
            &fixture.x,
        ),
        mean_slope,
        law: fit.latent_law_consumed.label(),
        calibrated: fit.latent_z_conditional_calibration.is_some(),
        certificate: match &fit.latent_law_consumed {
            gam::families::bms::LatentLawConsumed::EstimatedGaussianAdequate {
                residual: Some(certificate),
                ..
            }
            | gam::families::bms::LatentLawConsumed::EstimatedGlobalByResidual {
                residual: certificate,
                ..
            } => Some((certificate.excess_kl, certificate.null_p_value)),
            _ => None,
        },
    }
}

/// The marginal x-coefficient this fixture produces when the raw `z` axis is
/// anchored on `N(0, 1)`. Derived in the module header, not measured from an old
/// build, so the gate is a statement about the model.
fn gaussian_raw_axis_marginal_x_slope() -> f64 {
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let slope_raw = TRUE_SLOPE / residual_sd;
    let c_raw = (1.0 + slope_raw * slope_raw).sqrt();
    TRUE_BETA_X * c_true / c_raw - (TRUE_SLOPE * M_SHIFT / residual_sd) / c_raw
}

#[test]
fn conditional_latent_gate_returns_the_marginal_index() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let fixture = build_fixture();
    let shifted = fit_arm(&fixture, "z", None);
    let clean = fit_arm(&fixture, "zeta", None);
    let location_scale = fit_arm(&fixture, "z", Some("conditional-location-scale"));
    let gaussian_raw = gaussian_raw_axis_marginal_x_slope();
    let raw_axis_slope = TRUE_SLOPE / (1.0 - M_SHIFT * M_SHIFT).sqrt();

    eprintln!(
        "[2768] default z: law={} beta_x={:.4} slope={:.4} | default zeta: law={} beta_x={:.4} \
         slope={:.4} | location-scale z: law={} beta_x={:.4} slope={:.4} | truth beta_x={TRUE_BETA_X:.4} \
         slope(zeta axis)={TRUE_SLOPE:.4} slope(z axis)={raw_axis_slope:.4} | gaussian on raw z \
         beta_x={gaussian_raw:.4}",
        shifted.law,
        shifted.marginal_x_slope,
        shifted.mean_slope,
        clean.law,
        clean.marginal_x_slope,
        clean.mean_slope,
        location_scale.law,
        location_scale.marginal_x_slope,
        location_scale.mean_slope,
    );

    // 1. The conditional law moves on the shifted axis and not on the clean one.
    //    The shifted score is location-scale with a Gaussian residual by
    //    construction, so the moving-law certificate keeps the location-scale
    //    Gaussian law it fits first (gam#2926 diag15: on the survival form of this
    //    fixture that arm had the lowest held-out loss and exact risk from every
    //    anchor source), which calibrates the score onto the `ζ` axis.
    assert_eq!(
        shifted.law, "estimated-location-scale-gaussian",
        "the conditional law of z moves on the span at Corr(z, x) = {M_SHIFT}, n = {N}"
    );
    assert!(
        shifted.calibrated,
        "the location-scale Gaussian law fits m(x) on the shifted score"
    );
    // The law of an already conditionally standard score does not move on the span
    // and passes the adequacy screen, so the closed form's certificate decides: a
    // trigger-happy span test would make every clean fit local instead.
    let Some((clean_excess_kl, clean_null_p_value)) = clean.certificate else {
        panic!(
            "a conditionally standard score must pass the adequacy screen and carry the closed-form \
             certificate; got law={}",
            clean.law
        )
    };
    assert_eq!(
        clean.law,
        if clean_null_p_value.is_some_and(|p| p >= gam::families::bms::CLOSED_FORM_CERTIFICATE_ALPHA) {
            "estimated-gaussian-adequate"
        } else {
            "estimated-global-by-residual"
        },
        "the clean arm's law must follow its recorded null tail {clean_null_p_value:?} \
         against the design rate (D̂ = {clean_excess_kl:.4e})"
    );
    assert_eq!(location_scale.law, "conditional-location-scale");
    assert!(
        location_scale.calibrated,
        "the declared location-scale law must fit m(x) on this score"
    );

    // 2. Every arm returns the MARGINAL index, and the slope of its own axis.
    for (label, arm, slope) in [
        ("default z", &shifted, TRUE_SLOPE),
        ("default zeta", &clean, TRUE_SLOPE),
        ("location-scale z", &location_scale, TRUE_SLOPE),
    ] {
        assert!(
            (arm.marginal_x_slope - TRUE_BETA_X).abs() < 0.07,
            "{label} arm must return the marginal x-coefficient {TRUE_BETA_X}; got {:.4}",
            arm.marginal_x_slope
        );
        assert!(
            (arm.mean_slope - slope).abs() < 0.12,
            "{label} arm must return the slope of its own axis {slope:.4}; got {:.4}",
            arm.mean_slope
        );
    }

    // 3. The estimate must not depend on whether the shift was removed
    //    upstream, and must be nowhere near what a Gaussian anchor on the raw
    //    axis gives. This is the clause that goes red if the default ever stops
    //    anchoring on the conditional law.
    let arm_gap = (shifted.marginal_x_slope - clean.marginal_x_slope).abs();
    let to_gaussian_raw = (shifted.marginal_x_slope - gaussian_raw).abs();
    assert!(
        arm_gap < 0.06,
        "the fitted marginal index must not depend on which axis it was handed; arms \
         differ by {arm_gap:.4}"
    );
    assert!(
        to_gaussian_raw > 0.25,
        "fixture invariant: the default arm must be far from the Gaussian raw-axis value \
         {gaussian_raw:.4}, or this test cannot tell a working law from an absent one; \
         distance {to_gaussian_raw:.4}"
    );

    // 4. A Gaussian declaration on the shifted score is refused by name.
    match fit_from_formula("y ~ x", &fixture.dataset, &config("z", Some("gaussian"))) {
        Ok(_) => panic!("a Gaussian declaration on a conditionally shifted score must be refused"),
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("the Gaussian latent law was declared")
                    && message.contains("conditional mean or variance"),
                "the refusal must name the declaration and the moving moment; got {message}"
            );
        }
    }
}
