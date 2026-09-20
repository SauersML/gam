//! gam#3020: an explicit Duchon `length_scale=<number>` is a request, not a
//! seed. Every family must hand it back exactly as supplied in its resolved
//! (frozen) spec, and `length_scale=auto` must still be data-seeded, enrolled
//! as an outer κ coordinate and resolved by the fit.
//!
//! Before the fix the standard and survival paths enrolled an explicit scale as
//! a searched κ axis (treating the number as an initial value) while the
//! marginal-slope and location-scale paths held it fixed through their own
//! short-circuits, so the same formula meant two different models depending on
//! the family.
use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::basis::MaternLengthScale;
use gam_terms::smooth::{SmoothBasisSpec, TermCollectionSpec, spatial_term_supports_hyper_optimization};

const ROWS: usize = 300;
const PINNED: f64 = 0.7;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Columns `y, entry, exit, event, x1, x2, z`: a smooth 2-D signal
/// `f(x1, x2) = sin(πx1)·cos(πx2/2)` drives a Gaussian response `y`, a probit
/// binary response `yb` (through the latent `z`), and a log-normal event time.
fn dataset() -> EncodedDataset {
    let headers = ["y", "yb", "entry", "exit", "event", "x1", "x2", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut state = 0x3020u64;
    let rows = (0..ROWS)
        .map(|_| {
            let x1 = 2.0 * next_unit(&mut state) - 1.0;
            let x2 = 2.0 * next_unit(&mut state) - 1.0;
            let z = next_gauss(&mut state);
            let f = (std::f64::consts::PI * x1).sin() * (std::f64::consts::FRAC_PI_2 * x2).cos();
            let y = f + 0.3 * next_gauss(&mut state);
            let yb = u8::from(next_unit(&mut state) < normal_cdf(0.8 * f + 0.7 * z));
            let t = (0.5 * f + 0.5 * next_gauss(&mut state)).exp();
            let c = (0.4 + 2.0 * next_unit(&mut state)).exp();
            let (exit, event) = if t <= c { (t, 1u8) } else { (c, 0u8) };
            StringRecord::from(vec![
                format!("{y:.17e}"),
                yb.to_string(),
                "0".to_string(),
                format!("{exit:.17e}"),
                event.to_string(),
                format!("{x1:.17e}"),
                format!("{x2:.17e}"),
                format!("{z:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn duchon(length_scale: &str) -> String {
    format!("duchon(x1, x2, k=12, power=1, length_scale={length_scale})")
}

fn duchon_term(spec: &TermCollectionSpec) -> (usize, Option<MaternLengthScale>) {
    spec.smooth_terms
        .iter()
        .enumerate()
        .find_map(|(idx, term)| match &term.basis {
            SmoothBasisSpec::Duchon { spec, .. } => Some((idx, spec.length_scale)),
            _ => None,
        })
        .expect("resolved spec keeps the Duchon term")
}

/// The pinned fit returns `Fixed(PINNED)` bit-for-bit and does not enroll κ;
/// the learned fit returns a resolved `Auto` that is still enrolled, so a refit
/// or warm start keeps learning it.
fn assert_pinned_and_learned(family: &str, fit: impl Fn(&str) -> TermCollectionSpec) {
    let pinned = fit(&PINNED.to_string());
    let (idx, scale) = duchon_term(&pinned);
    assert_eq!(
        scale,
        Some(MaternLengthScale::fixed(PINNED)),
        "{family}: an explicit length_scale={PINNED} must come back unchanged"
    );
    assert!(
        !spatial_term_supports_hyper_optimization(&pinned, idx),
        "{family}: a pinned length_scale must not be an outer κ coordinate"
    );

    let learned = fit("auto");
    let (idx, scale) = duchon_term(&learned);
    match scale {
        Some(MaternLengthScale::Auto {
            resolved: Some(value),
        }) if value.is_finite() && value > 0.0 => {}
        other => panic!("{family}: length_scale=auto must resolve to a learned scale, got {other:?}"),
    }
    assert!(
        spatial_term_supports_hyper_optimization(&learned, idx),
        "{family}: a learned length_scale must stay an outer κ coordinate"
    );
}

#[test]
fn explicit_length_scale_is_pinned_in_the_standard_family_3020() {
    let data = dataset();
    assert_pinned_and_learned("standard", |ls| {
        let result = fit_from_formula(&format!("y ~ {}", duchon(ls)), &data, &FitConfig::default())
            .expect("standard fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard fit");
        };
        fit.resolvedspec
    });
}

#[test]
fn explicit_length_scale_is_pinned_in_the_gaussian_location_scale_family_3020() {
    let data = dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x1, k=6)".to_string()),
        ..FitConfig::default()
    };
    assert_pinned_and_learned("gaussian location-scale", |ls| {
        let result = fit_from_formula(&format!("y ~ {}", duchon(ls)), &data, &config)
            .expect("location-scale fit");
        let FitResult::GaussianLocationScale(fit) = result else {
            panic!("expected a Gaussian location-scale fit");
        };
        fit.fit.meanspec_resolved
    });
}

#[test]
fn explicit_length_scale_is_pinned_in_the_bernoulli_marginal_slope_family_3020() {
    let data = dataset();
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    assert_pinned_and_learned("bernoulli marginal-slope", |ls| {
        let result = fit_from_formula(&format!("yb ~ {}", duchon(ls)), &data, &config)
            .expect("marginal-slope fit");
        let FitResult::BernoulliMarginalSlope(fit) = result else {
            panic!("expected a Bernoulli marginal-slope fit");
        };
        fit.marginalspec_resolved
    });
}

#[test]
fn explicit_length_scale_is_pinned_in_the_survival_location_scale_family_3020() {
    let data = dataset();
    let config = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        ..FitConfig::default()
    };
    assert_pinned_and_learned("survival location-scale", |ls| {
        let result = fit_from_formula(
            &format!("Surv(entry, exit, event) ~ {}", duchon(ls)),
            &data,
            &config,
        )
        .expect("survival location-scale fit");
        let FitResult::SurvivalLocationScale(fit) = result else {
            panic!("expected a survival location-scale fit");
        };
        fit.fit.resolved_thresholdspec
    });
}
