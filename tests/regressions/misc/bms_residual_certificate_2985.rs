//! gam#2985 acceptance: a Bernoulli marginal-slope fit with a gam#2924 residual
//! repair block is certified on its joint `(z, r)` anchor.
//!
//! Given the score, the fit models the residual drive as `βᵀr | z ~ N(u z, v − u²)`
//! (`u = βᵀγ`, `v = βᵀΣ_rrβ`, `Σ₀₀ = 1`), so the row's anchor under any law of the
//! score is `Φ(ã + B z)` at `B = s(g + u)/τ`, `ã = α/τ`, `τ = √(1 + s²(v − u²))`.
//! Both latent-law certificates then run as they do without the block.
//!
//! 1. **Closed form.** On a Gaussian score the adequacy screen keeps the closed
//!    form, and its certificate is taken on the joint anchor: the fit records a
//!    certificate, not an uncertified reason. Its `D̂` agrees in sign with the same
//!    fit without the block, or the two differ by less than one standard error.
//! 2. **Moving law.** On a location-scale law of the score with `r` independent of
//!    `z`, the moving-law certificate chooses the same arm with the block as
//!    without it. The moving law fires the conditional calibration, so the
//!    score is a generated regressor; with `r` independent of `z` the joint
//!    covariance stays pooled, whose Murphy–Topel channel the block carries, so
//!    the block's fit corrects and publishes its covariance as the score-only
//!    fit does. Only the conditional `Σ(a)` withholds it, and that record
//!    survives the wire.
use csv::StringRecord;
use gam::estimate::CovarianceDeclined;
use gam::families::bms::{
    ClosedFormAnchorResidual, LatentLawConsumed, MovingLawArm, MovingLawCertificate,
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

/// The residual block's planted coefficients; `r` is independent of `z` and `x`.
const BETA: [f64; 2] = [0.3, -0.2];

fn config(with_block: bool) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        residual_columns: if with_block {
            vec!["r1".to_string(), "r2".to_string()]
        } else {
            Vec::new()
        },
        ..FitConfig::default()
    }
}

/// Rows `(y, z, x…, r1, r2)`: `z | x ~ N(mean(x), sd²)`, `r ~ N(0, I)` independent,
/// and `P(Y = 1 | x, z, r) = Φ(c(x)·q(x) + b(z − mean(x)) + βᵀr)` with
/// `c = √(1 + b²sd² + βᵀβ)`, so `E[Y | x] = Φ(q(x))`: the anchor holds on the
/// true joint law.
fn fixture(
    n: usize,
    seed: u64,
    covariates: usize,
    mean: impl Fn(&[f64]) -> f64,
    sd: f64,
    index: impl Fn(&[f64]) -> f64,
    drive: f64,
) -> gam::inference::data::EncodedDataset {
    let mut state = seed;
    let spread = (1.0 + drive * drive * sd * sd + BETA[0] * BETA[0] + BETA[1] * BETA[1]).sqrt();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<f64> = (0..covariates).map(|_| next_gauss(&mut state)).collect();
        let centre = mean(&x);
        let z = centre + sd * next_gauss(&mut state);
        let r = [next_gauss(&mut state), next_gauss(&mut state)];
        let eta = spread * index(&x) + drive * (z - centre) + BETA[0] * r[0] + BETA[1] * r[1];
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        let mut record = vec![y.to_string(), z.to_string()];
        record.extend(x.iter().map(|v| v.to_string()));
        record.extend(r.iter().map(|v| v.to_string()));
        rows.push(StringRecord::from(record));
    }
    let mut headers = vec!["y".to_string(), "z".to_string()];
    headers.extend((1..=covariates).map(|j| format!("x{j}")));
    headers.extend(["r1".to_string(), "r2".to_string()]);
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2985 fixture")
}

/// What a fit reports: the latent law it consumed, and its covariance or why it
/// withheld one.
struct Fitted {
    law: LatentLawConsumed,
    covariance_published: bool,
    covariance_declined: Option<CovarianceDeclined>,
}

fn fitted(formula: &str, data: &gam::inference::data::EncodedDataset, with_block: bool) -> Fitted {
    let result = fit_from_formula(formula, data, &config(with_block))
        .unwrap_or_else(|e| panic!("bernoulli marginal-slope fit (block={with_block}): {e}"));
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit");
    };
    let covariance_published = fit.fit.covariance_conditional.is_some();
    let covariance_declined = fit.fit.artifacts.covariance_declined.clone();
    Fitted {
        law: fit.latent_law_consumed,
        covariance_published,
        covariance_declined,
    }
}

fn closed_form_certificate(consumed: &LatentLawConsumed) -> ClosedFormAnchorResidual {
    match consumed {
        LatentLawConsumed::EstimatedGaussianAdequate {
            residual: Some(certificate),
            ..
        } => certificate.clone(),
        LatentLawConsumed::EstimatedGlobalByResidual { residual, .. } => residual.clone(),
        other => panic!(
            "a Gaussian score must carry the closed form's certificate; got {} ({:?})",
            other.label(),
            other.uncertified_reason()
        ),
    }
}

fn moving_law_certificate(consumed: &LatentLawConsumed) -> MovingLawCertificate {
    match consumed {
        LatentLawConsumed::EstimatedMovingLaw {
            certificate: Some(certificate),
            ..
        } => certificate.clone(),
        other => panic!(
            "a moving conditional law must be certified; got {} ({:?})",
            other.label(),
            other.uncertified_reason()
        ),
    }
}

#[test]
fn a_residual_repair_fit_takes_the_closed_form_certificate_on_its_joint_anchor_2985() {
    init_parallelism();
    let data = fixture(4_000, 0x2985_0000_0000_0001, 1, |_| 0.0, 1.0, |x| -0.3 + 0.4 * x[0], 0.6);
    let with_fit = fitted("y ~ x1", &data, true);
    let without_fit = fitted("y ~ x1", &data, false);
    // No calibration fires on a Gaussian score, so the score is not a generated
    // regressor and the block's covariance is published.
    for (fit, block) in [(&with_fit, true), (&without_fit, false)] {
        assert!(
            fit.covariance_published && fit.covariance_declined.is_none(),
            "an uncalibrated fit publishes its covariance (block={block}): declined {:?}",
            fit.covariance_declined
        );
    }
    let with = closed_form_certificate(&with_fit.law);
    let without = closed_form_certificate(&without_fit.law);
    // On a Gaussian score the anchors' noise is close to one shared mode Z, so
    // D̂ ≈ noise·(Z² − 2), with standard deviation about √2 times its noise energy.
    let standard_error = 2.0_f64.sqrt() * with.noise_energy.max(without.noise_energy);
    eprintln!(
        "[2985 closed form] with the block: D̂={:+.4e} (residual {:.4e}, noise {:.4e}, kept={}) | \
         without: D̂={:+.4e} (residual {:.4e}, noise {:.4e}, kept={}) | se {standard_error:.4e}",
        with.excess_kl,
        with.residual_energy,
        with.noise_energy,
        with.closed_form_chosen,
        without.excess_kl,
        without.residual_energy,
        without.noise_energy,
        without.closed_form_chosen,
    );
    assert!(
        with.excess_kl.signum() == without.excess_kl.signum()
            || (with.excess_kl - without.excess_kl).abs() <= standard_error,
        "the joint anchor's D̂ {:+.4e} must agree in sign with the score-only fit's {:+.4e}, or \
         within one standard error {standard_error:.4e}",
        with.excess_kl,
        without.excess_kl
    );
    let saved = gam::inference::model_payload_builders::fit_formula_to_payload(
        "y ~ x1".to_string(),
        &data,
        &config(true),
    )
    .unwrap_or_else(|e| panic!("a certified residual repair fit must save: {e}"));
    saved
        .latent_law_consumed
        .expect("a saved marginal-slope model records the law it consumed")
        .require_certified("a certified residual repair fit")
        .unwrap_or_else(|e| panic!("the saved record must be certified: {e}"));
}

#[test]
fn a_residual_repair_fit_certifies_the_same_moving_law_arm_as_without_the_block_2985() {
    init_parallelism();
    let data = fixture(
        10_000,
        0x2985_0000_0000_0002,
        2,
        |x| 0.5 * x[0] + 0.3 * x[1],
        0.66_f64.sqrt(),
        |x| -0.5 + 0.4 * x[0] - 0.3 * x[1],
        0.8,
    );
    let with_fit = fitted("y ~ x1 + x2", &data, true);
    let without_fit = fitted("y ~ x1 + x2", &data, false);
    // The moving law fires the conditional calibration, so the score is a
    // generated regressor. With `r` independent of `z` the block's joint
    // covariance is the pooled one, whose channel the correction carries: both
    // fits correct and publish their covariance.
    for (fit, block) in [(&with_fit, true), (&without_fit, false)] {
        assert!(
            fit.covariance_published && fit.covariance_declined.is_none(),
            "a calibrated fit corrects and publishes its covariance (block={block}): declined {:?}",
            fit.covariance_declined
        );
    }
    let with = moving_law_certificate(&with_fit.law);
    let without = moving_law_certificate(&without_fit.law);
    let describe = |certificate: &MovingLawCertificate| {
        certificate
            .arms
            .iter()
            .map(|s| format!("{:?} d={:+.3e} row_se={:.3e}", s.arm, s.difference, s.row_se))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    eprintln!(
        "[2985 moving law] with the block: fitted {:?} chosen {:?} ({}) || without: fitted {:?} \
         chosen {:?} ({})",
        with.fitted,
        with.chosen,
        describe(&with),
        without.fitted,
        without.chosen,
        describe(&without)
    );
    assert_eq!(with.fitted, without.fitted, "both fits start from the same arm");
    assert_eq!(
        with.chosen, without.chosen,
        "with r independent of z the joint anchor must choose the score-only arm"
    );
    assert_eq!(without.chosen, MovingLawArm::LocationScaleGaussian);
}

#[test]
fn a_withheld_residual_repair_covariance_survives_the_wire_2985() {
    let declined = CovarianceDeclined::BmsGeneratedRegressorResidualRepairChannelUnavailable {
        unavailable_channel: "the joint (z, r) covariance is the conditional Σ(a)".to_string(),
    };
    let explanation = declined.explain();
    assert!(
        explanation.contains("gam#2985") && explanation.contains("residual repair block"),
        "the explanation names the block and the issue: {explanation}"
    );
    let mut artifacts = gam::estimate::FitArtifacts::default();
    artifacts.covariance_declined = Some(declined.clone());
    let encoded = serde_json::to_string(&artifacts).expect("artifacts serialize");
    assert!(
        encoded.contains("bms-generated-regressor-residual-repair-channel-unavailable"),
        "the record is tagged by its reason: {encoded}"
    );
    let decoded: gam::estimate::FitArtifacts =
        serde_json::from_str(&encoded).expect("artifacts deserialize");
    assert_eq!(decoded.covariance_declined, Some(declined));
}
