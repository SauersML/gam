#![cfg(test)]
//! #3711: the latent interval warm start picks its surrogate from the data. A
//! positive-weight exact failure gives the right-censored-at-L surrogate an
//! interior optimum. Without one, the fit takes the lower-endpoint event
//! surrogate directly. The choice no longer depends on whether a first solve
//! returned `Err`. A zero-weight exact row is dormant, so it does not count as
//! a failure.

use super::*;
use crate::fit_orchestration::{FitConfig, FitRequest, materialize};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;

const N_ROWS: usize = 120;
const WEIBULL_SCALE: f64 = 2.0;
const WEIBULL_SHAPE: f64 = 1.3;
const FRAILTY_SIGMA: f64 = 0.5;
/// Inspection visits every half unit up to `LAST_VISIT`; later events are
/// bracketed by `(LAST_VISIT, HORIZON]`.
const VISIT_STEP: f64 = 0.5;
const LAST_VISIT: f64 = 6.0;
const HORIZON: f64 = 7.0;

/// SplitMix64 uniforms in `(0, 1)`.
struct DetRng {
    state: u64,
}

impl DetRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
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

/// Every row is bracketed `(L, R]` by the inspection grid, so the formula path
/// marks every row with the interval sentinel.
fn all_interval_rows() -> gam_data::EncodedDataset {
    let mut rng = DetRng { state: 0x3711_0834 };
    let mut records = Vec::with_capacity(N_ROWS);
    for _ in 0..N_ROWS {
        let frailty = (FRAILTY_SIGMA * rng.normal()).exp();
        let t = WEIBULL_SCALE * (-rng.uniform().ln() / frailty).powf(1.0 / WEIBULL_SHAPE);
        let (left, right) = if t > LAST_VISIT {
            (LAST_VISIT, HORIZON)
        } else {
            let visit = (t / VISIT_STEP).ceil().max(1.0);
            ((visit - 1.0) * VISIT_STEP, visit * VISIT_STEP)
        };
        records.push(StringRecord::from(vec![
            left.max(1e-6).to_string(),
            right.to_string(),
            "1".to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(
        vec!["L".to_string(), "R".to_string(), "event".to_string()],
        records,
    )
    .expect("encode the interval-censored rows")
}

/// Fit the all-interval fixture after `edit` rewrites the materialized spec.
fn fit_interval_fixture(
    edit: impl FnOnce(&mut LatentSurvivalTermSpec),
) -> Result<LatentSurvivalTermFitResult, FitFailure> {
    drop(gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
        gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
    )));
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(Box::new(
        gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator,
    )));
    let data = all_interval_rows();
    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed {
                sigma: FRAILTY_SIGMA,
            },
            loading: HazardLoading::Full,
        },
        ..FitConfig::default()
    };
    let model = materialize("SurvInterval(L, R, event) ~ 1", &data, &cfg)
        .expect("materialize the interval-censored fixture");
    let FitRequest::LatentSurvival(request) = model.request else {
        panic!("expected a latent survival request for survival_likelihood=latent");
    };
    let mut spec = request.spec;
    assert!(
        spec.event_target
            .iter()
            .all(|&code| code == LATENT_SURVIVAL_EVENT_INTERVAL),
        "the fixture brackets every row"
    );
    edit(&mut spec);
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(64 << 20)
        .build()
        .expect("survival worker pool");
    pool.install(|| fit_latent_survival_terms(request.data, spec, request.frailty, &request.options))
}

fn assert_finite_fit(fit: &LatentSurvivalTermFitResult, case: &str) {
    for (k, state) in fit.fit.block_states.iter().enumerate() {
        assert!(
            state.beta.iter().all(|v| v.is_finite()),
            "#3711 {case}: block {k} has a non-finite coefficient: {:?}",
            state.beta
        );
    }
}

/// With no exact failure at all, the lower-endpoint event surrogate seeds the
/// interval fit.
#[test]
fn an_all_interval_fit_seeds_from_the_lower_endpoint_surrogate_3711() {
    let fit = fit_interval_fixture(|spec| {
        assert!(!spec.weights.is_empty(), "the all-interval baseline must contain observed rows");
        assert!(
            spec.weights.iter().all(|&weight| weight.is_finite() && weight > 0.0),
            "the all-interval baseline must observe every interval with positive finite weight"
        );
    })
    .expect("#3711: an all-interval latent fit must converge from its surrogate seed");
    assert_finite_fit(&fit, "all-interval");
}

/// A zero-weight exact row is dormant and is not a failure. The fit still has
/// no positive-weight failure, so it takes the lower-endpoint event surrogate.
/// The old test `code != 0` counted this row as a failure and returned a hard
/// error.
#[test]
fn a_zero_weight_exact_row_is_not_a_failure_for_the_surrogate_choice_3711() {
    let fit = fit_interval_fixture(|spec| {
        spec.event_target[0] = 1;
        spec.weights[0] = 0.0;
    })
    .expect("#3711: a zero-weight exact row must not force the censored-at-L surrogate");
    assert_finite_fit(&fit, "zero-weight exact row");
}

/// A positive-weight exact failure gives the censored-at-L surrogate an
/// interior optimum, and the fit seeds from it.
#[test]
fn a_positive_weight_exact_failure_seeds_from_the_censored_surrogate_3711() {
    let fit = fit_interval_fixture(|spec| {
        spec.event_target[0] = 1;
    })
    .expect("#3711: a fit with a positive-weight exact failure must converge");
    assert_finite_fit(&fit, "positive-weight exact failure");
}
