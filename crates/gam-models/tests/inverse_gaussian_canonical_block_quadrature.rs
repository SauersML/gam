//! Inverse Gaussian on its canonical `1/μ²` link: the #784 block correction must resolve.
//!
//! The canonical inverse-Gaussian link, like the Gamma inverse link, is defined only on `η > 0`.
//! Along a penalized block axis the excess `ΔF(t)` is finite on an interval and infinite past
//! the point where some row's `η` reaches zero, so the block integrand
//! `e^{−ΔF}·1_feasible` jumps (or goes to a root) at that cut. A Gauss–Hermite rule on the whole
//! line converges only algebraically on such an integrand: the order search walked to the
//! largest representable order, 388, and refused the fit with
//! `BlockQuadratureCorrectionRefused … unresolved through the largest representable
//! Gauss–Hermite order`. The sweep's n = 100 `p1` inverse-Gaussian cell failed that way
//! (as did the multi-smooth `p5` / `te` cells, whose remaining outer-loop failure is the block
//! reselection #3154 owns).
//!
//! The block target now reports its feasibility cuts and the correction integrates on the
//! truncated normal, transporting each rule onto the feasible interval, where the integrand is
//! smooth and the rule converges geometrically. The fixture is the sweep's seed-0 cell,
//! exported from `bench/pygam_compare/worker.py::make_data`.
//!
//! The binary installs the production Laplace corrector, as `src/lib.rs` does for the wheel; the
//! shared suite installs none, and without one the block correction never runs.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};

const P1_N100: &str = include_str!("fixtures/inverse_gaussian_canonical_p1_n100.csv");

fn install_production_correctors() {
    drop(
        gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
            gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
        )),
    );
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(
        Box::new(gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator),
    ));
}

fn dataset(csv_text: &str) -> EncodedDataset {
    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fixture row"))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

fn assert_canonical_fit_certifies(label: &str, formula: &str, csv_text: &str) {
    install_production_correctors();
    let data = dataset(csv_text);
    let config = FitConfig {
        family: Some("inverse-gaussian".to_string()),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula(formula, &data, &config) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("[{label}] an additive inverse-Gaussian formula must produce a standard fit"),
        Err(error) => panic!(
            "[{label}] the canonical-link fit must resolve its block correction on the \
             feasible interval: {error}"
        ),
    };
    assert!(
        fit.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
        "[{label}] the certified smoothing parameters must be finite: {:?}",
        fit.fit.log_lambdas,
    );
}

#[test]
fn inverse_gaussian_canonical_single_smooth_resolves_its_block_correction() {
    assert_canonical_fit_certifies("p1 n=100", "y ~ s(x0)", P1_N100);
}
