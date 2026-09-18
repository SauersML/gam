//! gam#2926 acceptance: where the conditional law of the score moves on the
//! marginal-index span, the Bernoulli marginal-slope default fits the
//! location-scale Gaussian law and certifies it against the Gaussian,
//! location-scale empirical and local laws by their cross-fitted excess anchoring
//! loss, taking the simplest law within one paired standard error of the lowest.
//!
//! The model is `P(Y = 1 | x, z) = Φ(α(x) + b·z)` with `α` defined by the
//! anchoring equation on the TRUE law of `z | x`, `E[Φ(α(x) + b·z) | x] = Φ(q(x))`,
//! which for a normal mixture is `Σ_j p_j Φ((α + b·μ_j)/√(1 + b²σ_j²)) = Φ(q)`.
//!
//! 1. **A law whose shape moves is certified local.** `z | x` is a two-normal
//!    mixture whose weight on the upper component is `Φ(1.5·x)`: one normal at low
//!    `x`, bimodal near zero, the other normal at high `x`. No location-scale law
//!    follows that. At n = 10 000 the local law's exact risk under the true law is
//!    32 times the location-scale laws' lower, and its cross-fitted loss 15 paired
//!    standard errors lower (gam#2926 diag16), so the local law must clear the rule.
//! 2. **A location-scale law is certified on its Gaussian residual.** `z | x` is
//!    `N(0.5·x₁ + 0.3·x₂, 0.66)`, the law of synth_1e5_local; the local law is
//!    worse there by about 6 standard errors (diag16), so no re-solve happens.
//! 3. **The certificate does not depend on the worker pool.** The folds are fixed
//!    by the data and every sum runs in row order, so two fits on different pool
//!    sizes choose the same arm, and their losses differ by no more than the
//!    fits' own coefficients do between pools (#1045 holds those to 1e-7 of their
//!    scale): here at most 1e-6 of each difference's standard error.

use csv::StringRecord;
use gam::families::bms::{
    LatentLawConsumed, LatentMeasureKind, MovingLawArm, MovingLawCertificate,
};
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam::probability::normal_cdf(x)
}

/// Root of a strictly increasing function by bisection on `[-40, 40]`.
fn increasing_root(f: impl Fn(f64) -> f64, target: f64) -> f64 {
    let (mut low, mut high) = (-40.0_f64, 40.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if f(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The slope of the planted model on the raw score.
const DRIVE: f64 = 0.8;

/// `(weight, mean, sd)` of each normal component of `z | x`.
type Components = Vec<(f64, f64, f64)>;

/// The anchor `α` with `Σ_j p_j Φ((α + b·μ_j)/√(1 + b²σ_j²)) = Φ(q)`.
fn anchor(components: &Components, q: f64) -> f64 {
    increasing_root(
        |alpha| {
            components
                .iter()
                .map(|&(weight, mean, sd)| {
                    weight * normal_cdf((alpha + DRIVE * mean) / (1.0 + DRIVE * DRIVE * sd * sd).sqrt())
                })
                .sum()
        },
        normal_cdf(q),
    )
}

/// Draw `(x columns, z, y)` rows: `z | x` from `law(x)`, `y` from the anchored
/// model at the marginal index `q(x)`.
fn fixture(
    n: usize,
    seed: u64,
    covariates: usize,
    law: impl Fn(&[f64]) -> Components,
    index: impl Fn(&[f64]) -> f64,
) -> gam::inference::data::EncodedDataset {
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<f64> = (0..covariates).map(|_| next_gauss(&mut state)).collect();
        let components = law(&x);
        let pick = next_unit(&mut state);
        let mut cumulative = 0.0;
        let &(_, mean, sd) = components
            .iter()
            .find(|&&(weight, _, _)| {
                cumulative += weight;
                pick < cumulative
            })
            .unwrap_or_else(|| components.last().expect("a component"));
        let z = mean + sd * next_gauss(&mut state);
        let eta = anchor(&components, index(&x)) + DRIVE * z;
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        let mut record = vec![y.to_string(), z.to_string()];
        record.extend(x.iter().map(|v| v.to_string()));
        rows.push(StringRecord::from(record));
    }
    let mut headers = vec!["y".to_string(), "z".to_string()];
    headers.extend((1..=covariates).map(|j| format!("x{j}")));
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2926 moving-law fixture")
}

/// The shape-moving mixture: upper weight `Φ(1.5·x)`, means `−0.8 + 0.2·x` and
/// `1.3 + 0.2·x`, sds 0.45 and 0.35; `q(x) = −0.5 + 0.4·x`.
fn shape_moving(n: usize) -> gam::inference::data::EncodedDataset {
    fixture(
        n,
        0x2926_5A9E_0000_0001,
        1,
        |x| {
            let upper = normal_cdf(1.5 * x[0]);
            vec![
                (1.0 - upper, -0.8 + 0.2 * x[0], 0.45),
                (upper, 1.3 + 0.2 * x[0], 0.35),
            ]
        },
        |x| -0.5 + 0.4 * x[0],
    )
}

/// synth_1e5_local's law: `N(0.5·x₁ + 0.3·x₂, 0.66)`; `q = −0.5 + 0.4·x₁ − 0.3·x₂`.
fn location_scale(n: usize) -> gam::inference::data::EncodedDataset {
    fixture(
        n,
        0x2926_1055_0000_0001,
        2,
        |x| vec![(1.0, 0.5 * x[0] + 0.3 * x[1], 0.66_f64.sqrt())],
        |x| -0.5 + 0.4 * x[0] - 0.3 * x[1],
    )
}

fn config() -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        ..FitConfig::default()
    }
}

/// The fit's law label, its latent measure, and its moving-law certificate.
fn certified(
    formula: &str,
    data: &gam::inference::data::EncodedDataset,
) -> (&'static str, LatentMeasureKind, MovingLawArm, MovingLawCertificate) {
    let result = fit_from_formula(formula, data, &config())
        .unwrap_or_else(|e| panic!("bernoulli marginal-slope fit: {e}"));
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit");
    };
    let label = fit.latent_law_consumed.label();
    let LatentLawConsumed::EstimatedMovingLaw {
        arm,
        certificate: Some(certificate),
        ..
    } = fit.latent_law_consumed
    else {
        panic!("a moving conditional law must be certified; got {label}")
    };
    eprintln!(
        "[2926 moving law] {formula}: {label} | {}",
        certificate
            .arms
            .iter()
            .map(|s| format!(
                "{:?} loss={:.4e} d={:+.3e} row_se={:.3e} fold_se={:.3e}",
                s.arm, s.loss, s.difference, s.row_se, s.fold_se
            ))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    (label, fit.latent_measure, arm, certificate)
}

#[test]
fn a_law_whose_shape_moves_is_certified_local_2926() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();
    let (label, measure, arm, certificate) = certified("y ~ x1", &shape_moving(10_000));
    assert_eq!(label, "estimated-local");
    assert_eq!(arm, MovingLawArm::Local);
    assert!(
        matches!(measure, LatentMeasureKind::LocalEmpirical { .. }),
        "the re-solve must anchor on the local law"
    );
    assert_eq!(certificate.fitted, MovingLawArm::LocationScaleGaussian);
    assert_eq!(certificate.argmin, MovingLawArm::Local);
    assert_eq!(certificate.chosen, MovingLawArm::Local);
    for score in certificate.arms.iter().filter(|s| s.arm != MovingLawArm::Local) {
        assert!(
            score.difference > score.row_se,
            "the local law must beat the {:?} law by more than one paired standard error: {score:?}",
            score.arm
        );
    }
}

#[test]
fn a_location_scale_law_is_certified_on_its_gaussian_residual_2926() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();
    let (label, measure, arm, certificate) = certified("y ~ x1 + x2", &location_scale(10_000));
    assert_eq!(label, "estimated-location-scale-gaussian");
    assert_eq!(arm, MovingLawArm::LocationScaleGaussian);
    assert!(matches!(measure, LatentMeasureKind::StandardNormal));
    assert_eq!(certificate.fitted, MovingLawArm::LocationScaleGaussian);
    assert_eq!(certificate.chosen, MovingLawArm::LocationScaleGaussian);
    let local = certificate
        .arms
        .iter()
        .find(|s| s.arm == MovingLawArm::Local)
        .expect("the local arm is scored");
    assert!(
        local.difference > local.row_se,
        "the local law must lose on a location-scale law: {local:?}"
    );
}

#[test]
fn the_moving_law_certificate_does_not_depend_on_the_worker_pool_2926() {
    init_parallelism();
    let data = location_scale(3_000);
    let on_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to build a {threads}-thread pool: {e}"))
            .install(|| certified("y ~ x1 + x2", &data).3)
    };
    let wide = on_pool(rayon::current_num_threads().clamp(4, 8));
    let narrow = on_pool(2);
    assert_eq!(
        (wide.argmin, wide.chosen),
        (narrow.argmin, narrow.chosen),
        "the chosen arm must not depend on the pool"
    );
    assert_eq!(wide.rows, narrow.rows);
    // The certificate's noise scale: the largest paired standard error in it.
    let noise = wide.arms.iter().map(|s| s.row_se).fold(0.0_f64, f64::max);
    let floor = 1e-6 * noise;
    for (a, b) in wide.arms.iter().zip(narrow.arms.iter()) {
        assert_eq!(a.arm, b.arm);
        assert!(
            (a.loss - b.loss).abs() <= floor && (a.difference - b.difference).abs() <= floor,
            "{:?}: the loss moved between pools by more than 1e-6 of its standard error: \
             {a:?} vs {b:?}",
            a.arm
        );
    }
}

/// gam#2926 with a gam#2924 residual repair block: the fit anchors on the joint
/// `(z, r)` law, which the latent-law certificates do not evaluate. On a Gaussian
/// score whose law does not move, the default is the closed form, as without the
/// stack, recorded `gaussian-uncertified` with that reason and saved; a caller
/// asking for a certified fit is refused with the reason.
#[test]
fn a_residual_repair_fit_records_why_it_carries_no_certificate_2926() {
    init_parallelism();
    let n = 4_000;
    let slope: f64 = 0.6;
    let beta: [f64; 2] = [0.3, 0.0];
    let drive_scale = (1.0 + slope * slope + beta[0] * beta[0] + beta[1] * beta[1]).sqrt();
    let mut state = 0x2926_2924_0000_0001_u64;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = next_gauss(&mut state);
        let z = next_gauss(&mut state);
        let r = [next_gauss(&mut state), next_gauss(&mut state)];
        let q = -0.3 + 0.4 * x;
        let eta = drive_scale * q + slope * z + beta[0] * r[0] + beta[1] * r[1];
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z.to_string(),
            x.to_string(),
            r[0].to_string(),
            r[1].to_string(),
        ]));
    }
    let headers = ["y", "z", "x1", "r1", "r2"].iter().map(|s| s.to_string()).collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the #2926 residual fixture");
    let config = FitConfig {
        residual_columns: vec!["r1".to_string(), "r2".to_string()],
        ..config()
    };
    let payload = gam::inference::model_payload_builders::fit_formula_to_payload(
        "y ~ x1".to_string(),
        &data,
        &config,
    )
    .unwrap_or_else(|e| panic!("a residual repair fit must be saved uncertified: {e}"));
    let consumed = payload
        .latent_law_consumed
        .expect("a saved marginal-slope model records the law it consumed");
    assert_eq!(consumed.label(), "gaussian-uncertified");
    let reason = consumed
        .uncertified_reason()
        .expect("the uncertified record names why");
    assert!(
        reason.contains("residual repair") && reason.contains("(z, r)"),
        "the reason must name the residual block's joint law: {reason}"
    );
    let refusal = consumed
        .require_certified("a certified fit")
        .expect_err("a fit with no certificate is not a certified fit");
    assert!(
        refusal.contains(reason),
        "the refusal must carry the recorded reason: {refusal}"
    );
}
