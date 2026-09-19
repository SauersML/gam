//! #2903 acceptance witness: a bounded linear fit that SELECTS its smoothing λ
//! must publish the smoothing-corrected coefficient covariance with typed
//! provenance, so default-mode interval requests carry ρ uncertainty.
//!
//! The fit-side chain under test (gam-models
//! `fit_orchestration/drivers/design_construction.rs`): `BoundedLinearFamily`
//! declares its third information derivative on closed-form likelihoods, so the
//! custom-family lane declares an analytic outer ρ-Hessian and mints
//! `C = A·V_ρ·Aᵀ` on the latent coefficients, and
//! `fit_bounded_term_collection_with_design` pushes it to user scale and
//! publishes `Vp = Vb + C`.
//!
//! DGP: deterministic Poisson counts with log mean `0.2 + 0.6·x + 0.8·sin(2πz)`,
//! `x ∈ [0, 1]` entering through `bounded(x, min=0, max=1)` and the curved `z`
//! effect through `s(z)`. The Poisson log link has a closed-form `W'''` and a
//! unit dispersion, so the correction is judged without a scale estimate.
//!
//! Assertions:
//! 1. the fit selected λ, and the corrected covariance and its typed method are
//!    on the fit with an interior rank of at least one;
//! 2. the corrected covariance matches the conditional dimensions and the
//!    top-level mirror;
//! 3. `C = V_c − V_cond` never shrinks a variance, and strictly widens at least
//!    one;
//! 4. symmetry and finiteness of the corrected matrix.
use csv::StringRecord;
use gam::model_types::SmoothingCorrectionMethod;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use std::sync::Once;

const N: usize = 400;

/// Routes the fit's Info lines (the certificate and the `[smoothing-correction]`
/// outcome) to stderr, so a red run says why the corrected covariance is absent.
struct StderrInfoLogger;

impl log::Log for StderrInfoLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
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

    /// Poisson draw by CDF inversion; the means here stay below 6.
    fn next_poisson(&mut self, mean: f64) -> u32 {
        let u = self.next_u01();
        let mut k = 0u32;
        let mut pmf = (-mean).exp();
        let mut cdf = pmf;
        while u > cdf && k < 200 {
            k += 1;
            pmf *= mean / f64::from(k);
            cdf += pmf;
        }
        k
    }
}

fn build_poisson_frame() -> gam::data::EncodedDataset {
    let mut rng = Lcg::new(0x2903_B0DD_5EED_0001);
    let headers = ["y", "x", "z"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let x = rng.next_u01();
        let z = (i as f64 + 0.5) / N as f64;
        let eta = 0.2 + 0.6 * x + 0.8 * (std::f64::consts::TAU * z).sin();
        let y = rng.next_poisson(eta.exp());
        rows.push(StringRecord::from(vec![
            format!("{y}"),
            format!("{x:.17e}"),
            format!("{z:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode bounded Poisson frame")
}

#[test]
fn bounded_fit_publishes_smoothing_corrected_covariance_2903() {
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
    let data = build_poisson_frame();
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ bounded(x, min=0, max=1) + s(z)", &data, &cfg)
        .expect("bounded Poisson fit with a smooth");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for the bounded Poisson formula");
    };
    let fit = &fit.fit;
    let selected = fit
        .lambdas
        .iter()
        .map(|lambda| format!("{lambda:.6e}"))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "#2903 bounded selected lambdas=[{selected}] outer_iterations={}",
        fit.outer_iterations
    );

    // (1) The fit selected λ; the corrected covariance and its typed provenance
    // are present.
    assert!(
        !fit.lambdas.is_empty() && fit.outer_iterations > 0,
        "#2903: the bounded fit must select its smoothing lambda"
    );
    let conditional = fit
        .beta_covariance()
        .expect("#2903: a bounded fit with inference on carries the conditional covariance");
    let corrected = fit.beta_covariance_corrected().expect(
        "#2903: a lambda-selecting bounded fit must carry the smoothing-corrected covariance",
    );
    match fit
        .smoothing_correction_method()
        .expect("#2903: the corrected covariance must carry its typed method provenance")
    {
        SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
            active_rank,
            rho_dimension,
        } => {
            assert_eq!(
                rho_dimension,
                fit.lambdas.len(),
                "#2903: the correction's rho dimension must be the selected lambda count"
            );
            assert!(
                active_rank >= 1 && active_rank <= rho_dimension,
                "#2903: identified interior rank {active_rank} must be within 1..={rho_dimension}; \
                 a zero rank means every smoothing coordinate railed and the fixture is vacuous",
            );
        }
        other => panic!("#2903: expected FirstOrderIdentifiedSubspace provenance, got {other:?}"),
    }

    // (2) Shape.
    assert_eq!(
        corrected.dim(),
        conditional.dim(),
        "#2903: corrected covariance must match the conditional dimensions"
    );

    // (3) The correction is PSD, so no corrected variance shrinks, and a selected
    // λ on finite data carries genuine uncertainty, so at least one variance
    // strictly grows.
    let p = corrected.nrows();
    let mut any_strict_growth = false;
    for i in 0..p {
        let vc = corrected[[i, i]];
        let v0 = conditional[[i, i]];
        assert!(
            vc >= v0 - 1e-10 * (1.0 + v0.abs()),
            "#2903: corrected variance must not shrink below conditional at coordinate {i}: \
             corrected {vc:.6e} vs conditional {v0:.6e}"
        );
        if vc > v0 * (1.0 + 1e-9) + 1e-14 {
            any_strict_growth = true;
        }
    }
    assert!(
        any_strict_growth,
        "#2903: the rho-uncertainty inflation must strictly widen at least one coefficient variance"
    );

    // (4) Symmetry and finiteness.
    for i in 0..p {
        for j in 0..p {
            let v = corrected[[i, j]];
            assert!(v.is_finite(), "#2903: corrected[{i},{j}] must be finite");
            assert!(
                (v - corrected[[j, i]]).abs() <= 1e-9 * (1.0 + v.abs()),
                "#2903: corrected covariance must be symmetric at [{i},{j}]"
            );
        }
    }
}
