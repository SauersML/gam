//! #2912 acceptance witness — a single-cause survival transformation fit that
//! SELECTS its time-smoothing λ must publish the smoothing-corrected
//! coefficient covariance with typed provenance, so interval requests at the
//! default covariance mode stop refusing single-cause models with "saved model
//! does not contain smoothing-corrected covariance".
//!
//! The fit-side chain under test (gam-models `fit_orchestration/fit.rs`): the
//! selector declares the analytic survival LAML ρ-Hessian, the terminal mint
//! evaluates it once, and `survival_unified_fit_result` publishes
//! `beta_covariance_corrected = V_cond + C`, `C = A·V_ρ·Aᵀ`, `A = V_cond·U`,
//! `U[:, k] = λ_k S_k β̂` over block k, through the same
//! `first_order_smoothing_correction` the custom-family mint uses.
//!
//! DGP: deterministic right-censored Gompertz hazard `h(t) = a·exp(b·t + η)`.
//! Its log cumulative hazard `log(a/b) + log(exp(b·t) − 1) + η` is curved in
//! `log t`, so the time block has real curvature to keep; a Weibull baseline is
//! affine there and drives its λ onto the rail.
//!
//! Assertions:
//! 1. the fit selected λ, and the corrected covariance and its typed method
//!    are on the fit with an interior rank of at least one;
//! 2. corrected matches the conditional covariance dimensions;
//! 3. `C = V_c − V_cond` never shrinks a variance, and strictly widens at least
//!    one;
//! 4. symmetry and finiteness of the corrected matrix.

use csv::StringRecord;
use gam::model_types::SmoothingCorrectionMethod;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use std::sync::Once;

const N: usize = 600;

/// Routes the fit's Info lines (the certificate and the `[smoothing-correction]`
/// outcome) to stderr, so a red run says why the corrected covariance is absent.
struct StderrInfoLogger;

impl log::Log for StderrInfoLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}

static LOGGER: StderrInfoLogger = StderrInfoLogger;
static INIT_LOGGER: Once = Once::new();

/// Deterministic LCG uniform stream (no RNG crate dependency).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_u01().max(1e-12);
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn build_gompertz_frame() -> gam::data::EncodedDataset {
    const RATE: f64 = 0.05;
    const GROWTH: f64 = 0.9;
    const SLOPE: f64 = 0.7;
    let mut rng = Lcg::new(0x2912_5EED_0BAD_F00D);

    let headers = ["t", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_normal();
        let eta = SLOPE * xi;
        let u = rng.next_u01().clamp(1e-12, 1.0 - 1e-6);
        // Inverse of S(t) = exp(−(a/b)·exp(η)·(exp(b·t) − 1)).
        let t_lat = (1.0 - GROWTH * u.ln() / (RATE * eta.exp())).ln() / GROWTH;
        let cens = (-rng.next_u01().max(1e-12).ln() * 6.0).min(12.0);
        let exit = t_lat.min(cens);
        let event = if t_lat <= cens { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            format!("{exit:.17e}"),
            format!("{event:.1}"),
            format!("{xi:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode Gompertz survival frame")
}

#[test]
fn survival_transformation_fit_publishes_smoothing_corrected_covariance_2912() {
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Debug);
        }
    });
    let data = build_gompertz_frame();
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 6,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ x + survmodel(spec=net)", &data, &cfg)
        .expect("Royston-Parmar transformation fit on the synthetic Gompertz cohort");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };
    let fit = &fit.fit;
    eprintln!(
        "#2912 selected lambdas={:?} outer_iterations={}",
        fit.lambdas.to_vec(),
        fit.outer_iterations
    );

    // (1) The fit selected λ; the corrected covariance and its typed provenance
    // are present.
    assert!(
        !fit.lambdas.is_empty() && fit.outer_iterations > 0,
        "#2912: the transformation fit must select its time-smoothing lambda"
    );
    let conditional = fit
        .covariance_conditional
        .as_ref()
        .expect("#2912: a converged transformation fit carries the conditional covariance");
    let corrected = fit.beta_covariance_corrected().expect(
        "#2912: a lambda-selecting single-cause fit must carry the smoothing-corrected \
         covariance so default-mode interval requests are served",
    );
    match fit
        .smoothing_correction_method()
        .expect("#2912: the corrected covariance must carry its typed method provenance")
    {
        SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
            active_rank,
            rho_dimension,
        } => {
            assert_eq!(
                rho_dimension,
                fit.lambdas.len(),
                "#2912: the correction's rho dimension must be the selected lambda count"
            );
            assert!(
                active_rank >= 1 && active_rank <= rho_dimension,
                "#2912: identified interior rank {active_rank} must be within 1..={rho_dimension}; \
                 a zero rank means every smoothing coordinate railed and the fixture is vacuous",
            );
        }
        other => panic!("#2912: expected FirstOrderIdentifiedSubspace provenance, got {other:?}"),
    }

    // (2) Shape.
    assert_eq!(
        corrected.dim(),
        conditional.dim(),
        "#2912: corrected covariance must match the conditional dimensions"
    );

    // (3) The correction is PSD, so no corrected variance shrinks, and REML
    // smoothing parameters on finite data carry genuine uncertainty, so at least
    // one variance strictly grows.
    let p = corrected.nrows();
    let mut any_strict_growth = false;
    for i in 0..p {
        let vc = corrected[[i, i]];
        let v0 = conditional[[i, i]];
        assert!(
            vc >= v0 - 1e-10 * (1.0 + v0.abs()),
            "#2912: corrected variance must not shrink below conditional at coordinate {i}: \
             corrected {vc:.6e} vs conditional {v0:.6e}"
        );
        if vc > v0 * (1.0 + 1e-9) + 1e-14 {
            any_strict_growth = true;
        }
    }
    assert!(
        any_strict_growth,
        "#2912: the rho-uncertainty inflation must strictly widen at least one coefficient \
         variance"
    );

    // (4) Symmetry and finiteness.
    for i in 0..p {
        for j in 0..p {
            let v = corrected[[i, j]];
            assert!(v.is_finite(), "#2912: corrected[{i},{j}] must be finite");
            assert!(
                (v - corrected[[j, i]]).abs() <= 1e-9 * (1.0 + v.abs()),
                "#2912: corrected covariance must be symmetric at [{i},{j}]"
            );
        }
    }
}
