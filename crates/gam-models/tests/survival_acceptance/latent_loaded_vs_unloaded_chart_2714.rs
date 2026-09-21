//! #2714, R-free: a loaded/unloaded latent survival fit selects its
//! Gompertz-Makeham baseline together with ρ on the one LAML criterion.
//!
//! A fully loaded latent fit already selects its baseline chart with ρ. A split
//! also moves the Makeham background, which no row primary carries, so its
//! `ln m` chart axis is served by the background-share jet. This fixture drives
//! that route end to end on data where the background genuinely competes with the
//! frailty-scaled hazard. The fit must come back, and its selected background
//! scale must have left the seed the workflow starts from.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_models::survival::construction::{
    SurvivalBaselineTarget, initial_survival_baseline_config_for_fit,
};
use gam_models::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use ndarray::Array1;

const N_ROWS: usize = 160;
const TRUE_BETA: f64 = 0.7;
const TRUE_SIGMA: f64 = 0.5;
const TRUE_RATE: f64 = 0.01;
const TRUE_SHAPE: f64 = 0.08;
const TRUE_MAKEHAM: f64 = 0.02;
const CENSOR_SPAN: f64 = 60.0;

/// SplitMix64 uniforms in `(0, 1)`, so the rows are bit-identical on every run.
struct DetRng {
    state: u64,
}

impl DetRng {
    fn next_u64(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.state)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn latent_loaded_vs_unloaded_fit_selects_its_background_with_rho_2714() {
    gam_runtime::test_support::install_diagnostic_logger();
    super::initialize_cpu_fitting();

    // The hazard is `m + z·rate·e^{shape·t}·e^{βx}`: an exponential background
    // that no frailty modifies, competing with a frailty-scaled Gompertz hazard.
    // Competing hazards add, so the event time is the first of the two times.
    let mut rng = DetRng { state: 0x2714_0913 };
    let mut records = Vec::with_capacity(N_ROWS);
    let mut age_exit = Array1::<f64>::zeros(N_ROWS);
    let mut events = 0usize;
    for row in 0..N_ROWS {
        let x = rng.uniform();
        let frailty = (TRUE_SIGMA * rng.normal()).exp();
        let background_time = -rng.uniform().ln() / TRUE_MAKEHAM;
        let multiplier = TRUE_RATE * frailty * (TRUE_BETA * x).exp();
        let loaded_time = (1.0 + TRUE_SHAPE * -rng.uniform().ln() / multiplier).ln() / TRUE_SHAPE;
        let censor_time = CENSOR_SPAN * (0.5 + rng.uniform());
        let event_time = background_time.min(loaded_time);
        let (time, status) = if event_time <= censor_time {
            events += 1;
            (event_time, 1.0)
        } else {
            (censor_time, 0.0)
        };
        age_exit[row] = time;
        records.push(StringRecord::from(vec![
            time.to_string(),
            status.to_string(),
            x.to_string(),
        ]));
    }
    assert!(
        events > N_ROWS / 4 && events < N_ROWS,
        "precondition: the fixture must carry both events and censoring, got {events} events in {N_ROWS} rows"
    );
    let data = encode_recordswith_inferred_schema(
        vec!["time".to_string(), "status".to_string(), "x".to_string()],
        records,
    )
    .expect("encode the loaded/unloaded survival rows");

    let seed = initial_survival_baseline_config_for_fit(
        "gompertz-makeham",
        None,
        None,
        None,
        None,
        &age_exit,
    )
    .expect("the workflow's Gompertz-Makeham seed");
    let seed_makeham = seed
        .makeham
        .expect("a Gompertz-Makeham seed carries a makeham rate");

    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "gompertz-makeham".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma: TRUE_SIGMA },
            loading: HazardLoading::LoadedVsUnloaded,
        },
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, status) ~ x", &data, &cfg)
        .expect("#2714: the loaded/unloaded latent fit must converge and return a fit");
    let FitResult::LatentSurvival(fit) = result else {
        panic!("expected a LatentSurvival fit result for survival_likelihood=latent");
    };

    let fitted = &fit.baseline_config;
    assert!(
        matches!(fitted.target, SurvivalBaselineTarget::GompertzMakeham),
        "#2714: a loaded/unloaded fit must keep its Gompertz-Makeham baseline"
    );
    let fitted_makeham = fitted
        .makeham
        .expect("#2714: the fitted Gompertz-Makeham baseline must carry its makeham rate");
    assert!(
        fitted_makeham.is_finite() && fitted_makeham > 0.0,
        "#2714: the selected makeham rate must be finite and positive, got {fitted_makeham}"
    );
    assert!(
        fitted_makeham.to_bits() != seed_makeham.to_bits(),
        "#2714: the background scale never left its seed {seed_makeham}: the ln m chart axis was not selected"
    );
    eprintln!(
        "[2714] loaded/unloaded latent fit: makeham={fitted_makeham} (seed {seed_makeham}, truth {TRUE_MAKEHAM}), rate={} (truth {TRUE_RATE}), shape={} (truth {TRUE_SHAPE}), events={events}/{N_ROWS}",
        fitted.rate.unwrap_or(f64::NAN),
        fitted.shape.unwrap_or(f64::NAN),
    );
}
