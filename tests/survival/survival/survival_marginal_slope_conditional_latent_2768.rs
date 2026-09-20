//! gam#2768 — the survival marginal-slope must run the same automatic latent
//! measure gate the Bernoulli marginal-slope has run since #905.
//!
//! # The estimand, and what a conditional shift does to it
//!
//! The survival marginal-slope row index is `η = q·c(g) + s(g)·z` with
//! `c = √(1 + s²)`, and the whole point of that parameterisation is the identity
//!
//! ```text
//!     E_z[Φ(q·√(1+b²) + b·z)] = Φ(q)     for   z | C ~ N(0, 1),
//! ```
//!
//! which is what makes `q` the **marginal** index. The identity is conditional
//! on `C`: it needs `z | C ~ N(0,1)`, not merely `z ~ N(0,1)`.
//!
//! This fixture builds a latent score that is **exactly standard normal
//! marginally** and conditionally shifted:
//!
//! ```text
//!     x, ζ ~ N(0,1) independent,     z = m·x + √(1−m²)·ζ
//! ```
//!
//! so `z ~ N(0,1)`, `Var(z) = 1`, and `E[z | x] = m·x`. The outcome is generated
//! from the marginal-slope model **on ζ**, so the truth is a marginal index
//! `q(t,x) = γ₀ + γ₁·log t + β_x·x` and a constant slope `b` on ζ.
//!
//! A fit that consumes the raw `z` axis is estimating a different model. Writing
//! `ζ = (z − m x)/√(1−m²)`:
//!
//! ```text
//!     η = q·c(b) + b·ζ
//!       = [q·c(b) − (b·m/√(1−m²))·x] + (b/√(1−m²))·z
//! ```
//!
//! so its slope becomes `b′ = b/√(1−m²)` and its marginal x-coefficient becomes
//! `β_x·c(b)/c(b′) − (b·m/√(1−m²))/c(b′)` — the `b(C)·m(C)` leakage, landing
//! squarely in the influence channel `q`. With the constants below that is
//! `0.195` against a truth of `0.500`: a **61% attenuation**, not a rounding
//! difference. The pooled marginal gate cannot see any of it (z is exactly
//! N(0,1)) and no monotone transform of the marginal law can remove it.
//!
//! A fit that anchors on the CONDITIONAL law of `z` — the default since gam#2926 —
//! is estimating the same model: `z | x ~ N(m·x, 1−m²)` is the location-scale
//! Gaussian law its moving-law certificate keeps here, which fits `m(x)` and
//! `v(x)`, anchors `ζ` in closed form, and has slope `b` on that axis.
//!
//! # The gates
//!
//! Three arms of the same data: the default on the shifted `z`, the default on
//! the conditionally standardised `ζ` the outcome was generated from, and the
//! declared conditional location-scale law on `z`. The assertions are
//!
//!   1. the default on `z` is certified on the location-scale Gaussian law, the
//!      fixture's law, which calibrates `z` onto `ζ`, and the default on `ζ` on
//!      one global law (the span test is neither asleep nor trigger-happy);
//!   2. every arm recovers `β_x`, and the slope of its own axis;
//!   3. the arms agree with each other far more tightly than any agrees with the
//!      Gaussian raw-axis value — this is the clause that goes red if the
//!      survival path ever stops anchoring on the conditional law.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam::utils::splitmix64;
use ndarray::Array1;

const N: usize = 3_000;
/// Conditional correlation `Corr(z, x)`; also the conditional-mean slope.
const M_SHIFT: f64 = 0.5;
/// True conditional slope on the standardised latent score.
const TRUE_SLOPE: f64 = 0.6;
/// True marginal-index coefficient on `x`.
const TRUE_BETA_X: f64 = 0.5;
/// True marginal-index intercept and `log t` coefficient.
const TRUE_INTERCEPT: f64 = -1.0;
const TRUE_LOG_TIME: f64 = 0.8;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `Φ⁻¹` by bisection on the shared `normal_cdf`. Deterministic, and accurate
/// well past what the fixture needs; the fit is what is under test, not the
/// quantile.
fn standard_normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gam::probability::normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

struct Fixture {
    dataset: gam::inference::data::EncodedDataset,
    x: Array1<f64>,
}

/// Columns: `time`, `event`, `z` (conditionally shifted), `zeta` (the axis the
/// outcome was generated on), `x`.
fn build_fixture() -> Fixture {
    let headers = ["time", "event", "z", "zeta", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2768_5EED_C0FF_EE01;
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut x_values = Array1::<f64>::zeros(N);
    for row in 0..N {
        let x = next_gauss(&mut state);
        let zeta = next_gauss(&mut state);
        let z = M_SHIFT * x + residual_sd * zeta;
        // Invert `Φ(η) = u` for the event time: the model is a probit
        // transformation model, `Φ⁻¹(F(t | x, ζ)) = q(t, x)·c + b·ζ`.
        let eta_target = standard_normal_quantile(next_unit(&mut state));
        let log_t = ((eta_target - TRUE_SLOPE * zeta) / c_true
            - TRUE_INTERCEPT
            - TRUE_BETA_X * x)
            / TRUE_LOG_TIME;
        let event_time = log_t.exp();
        // Independent administrative-style censoring, spread so no single exit
        // time carries a large tie mass.
        let censor_time = 1.5 + 3.0 * next_unit(&mut state);
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        x_values[row] = x;
        rows.push(StringRecord::from(vec![
            time.max(1e-6).to_string(),
            event.to_string(),
            z.to_string(),
            zeta.to_string(),
            x.to_string(),
        ]));
    }

    Fixture {
        dataset: encode_recordswith_inferred_schema(headers, rows)
            .expect("encode the #2768 conditional-latent survival fixture"),
        x: x_values,
    }
}

struct ArmSummary {
    /// Slope of the fitted MARGINAL linear predictor on `x`, read by projection
    /// rather than from a coefficient: the marginal block carries whatever
    /// identifiability chart the frozen joint build chose, so the coefficient
    /// index is not a stable contract but `∂(X_m β_m)/∂x` is.
    marginal_x_slope: f64,
    /// Weighted-mean fitted conditional slope on the latent score.
    mean_slope: f64,
    /// The latent law the fit consumed.
    law: &'static str,
    /// Whether the fit replaced the score by a calibrated one.
    calibrated: bool,
    /// `D̂` of the closed-form certificate, when the adequacy screen passed.
    /// The closed-form certificate's `(D̂, null tail)`, when one was taken.
    certificate: Option<(f64, Option<f64>)>,
}

fn slope_on_x(values: &Array1<f64>, x: &Array1<f64>) -> f64 {
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

/// The fit shape both arms use. Smooth on both channels because the
/// marginal-slope joint Newton is measurably better conditioned there: an
/// intercept-only slope block leaves the joint Hessian with a rank-1 block
/// whose null-space shrinkage direction is the whole block, and every candidate
/// rho seed's inner solve then stalls at its cycle budget (measured on this very
/// fixture before the shape was changed). The truth is linear in `x` and so lies
/// inside both smooths' spans, which is what lets the projection below read it
/// back without assuming a coefficient index.
const MARGINAL_FORMULA: &str = "Surv(time, event) ~ smooth(x)";
const SLOPE_FORMULA: &str = "smooth(x)";

fn fit_arm(fixture: &Fixture, z_column: &str, latent_measure: Option<&str>) -> ArmSummary {
    let cfg = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some(z_column.to_string()),
        slope_formula: Some(SLOPE_FORMULA.to_string()),
        baseline_target: "linear".to_string(),
        latent_measure: latent_measure.map(str::to_string),
        ..FitConfig::default()
    };
    let result = fit_from_formula(MARGINAL_FORMULA, &fixture.dataset, &cfg)
        .unwrap_or_else(|e| panic!("survival marginal-slope fit on z_column={z_column}: {e}"));
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result for z_column={z_column}");
    };

    let beta_marginal = &fit.fit.blocks[1].beta;
    let marginal_eta = fit.marginal_design.design.dot(beta_marginal);
    let beta_slope = &fit.fit.blocks[2].beta;
    let slope_eta = fit.slope_design.design.dot(beta_slope);
    let mean_slope = fit.baseline_slope
        + slope_eta.iter().sum::<f64>() / slope_eta.len() as f64;

    ArmSummary {
        marginal_x_slope: slope_on_x(&marginal_eta, &fixture.x),
        mean_slope,
        law: fit.latent_law_consumed.label(),
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
        calibrated: fit.latent_z_calibrations.first().is_some_and(|calibration| {
            !matches!(
                calibration,
                gam::families::bms::LatentMeasureCalibration::None
            )
        }),
    }
}

/// The marginal x-coefficient this fixture produces when the raw `z` axis is
/// anchored on `N(0, 1)` — the value the survival path returned before gam#2768.
/// Derived in the module header, not measured, so the gate is a statement about
/// the model rather than a snapshot of an old build.
fn uncalibrated_marginal_x_slope() -> f64 {
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let slope_raw = TRUE_SLOPE / residual_sd;
    let c_raw = (1.0 + slope_raw * slope_raw).sqrt();
    TRUE_BETA_X * c_true / c_raw - (TRUE_SLOPE * M_SHIFT / residual_sd) / c_raw
}

#[test]
fn survival_marginal_slope_removes_the_conditional_latent_shift() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let fixture = build_fixture();
    let shifted = fit_arm(&fixture, "z", None);
    let clean = fit_arm(&fixture, "zeta", None);
    let location_scale = fit_arm(&fixture, "z", Some("conditional-location-scale"));
    let uncalibrated = uncalibrated_marginal_x_slope();
    let raw_axis_slope = TRUE_SLOPE / (1.0 - M_SHIFT * M_SHIFT).sqrt();

    eprintln!(
        "[2768] default z: law={} beta_x={:.4} slope={:.4} | default zeta: law={} beta_x={:.4} \
         slope={:.4} | location-scale z: law={} beta_x={:.4} slope={:.4} | truth \
         beta_x={TRUE_BETA_X:.4} slope(zeta axis)={TRUE_SLOPE:.4} slope(z axis)={raw_axis_slope:.4} \
         | gaussian on raw z beta_x={uncalibrated:.4}",
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
    //    Gaussian law it fits first (gam#2926 diag15: on this fixture that arm had
    //    the lowest held-out loss and exact risk from every anchor source), which
    //    calibrates the score onto the `ζ` axis.
    assert_eq!(
        shifted.law, "estimated-location-scale-gaussian",
        "the conditional law of z moves on the span at a {M_SHIFT} conditional correlation, n={N}"
    );
    assert!(
        shifted.calibrated,
        "the location-scale Gaussian law fits m(x) on the shifted score"
    );
    // The law of an already conditionally standard score must not be local, and it
    // passes the adequacy screen, so the closed form's certificate decides: a
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

    // 2. Every arm recovers the marginal index and the slope of its own axis.
    for (label, arm, slope) in [
        ("default z", &shifted, TRUE_SLOPE),
        ("default zeta", &clean, TRUE_SLOPE),
        ("location-scale z", &location_scale, TRUE_SLOPE),
    ] {
        assert!(
            (arm.marginal_x_slope - TRUE_BETA_X).abs() < 0.06,
            "{label} arm must recover the marginal x-coefficient {TRUE_BETA_X}; got {:.4}",
            arm.marginal_x_slope
        );
        assert!(
            (arm.mean_slope - slope).abs() < 0.08,
            "{label} arm must recover the slope of its own axis {slope:.4}; got {:.4}",
            arm.mean_slope
        );
    }

    // 3. The arms agree with each other far more tightly than any agrees with
    //    the Gaussian raw-axis value. This is the clause that goes red if the
    //    survival path ever stops anchoring on the conditional law.
    let arm_gap = (shifted.marginal_x_slope - clean.marginal_x_slope)
        .abs()
        .max((location_scale.marginal_x_slope - clean.marginal_x_slope).abs());
    let uncalibrated_gap = (clean.marginal_x_slope - uncalibrated).abs();
    assert!(
        arm_gap < 0.05,
        "the fitted marginal index must not depend on whether the conditional shift was \
         removed upstream; arms differ by {arm_gap:.4}"
    );
    assert!(
        uncalibrated_gap > 6.0 * arm_gap.max(1e-3),
        "fixture invariant: the uncalibrated axis must be far from the truth, otherwise this \
         test cannot distinguish a working gate from an absent one; uncalibrated \
         gap={uncalibrated_gap:.4}, arm gap={arm_gap:.4}"
    );
}
