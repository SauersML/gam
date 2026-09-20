//! Scaled-t on a tensor smooth: a rejected Anderson trial must not inflate the P-IRLS damping.
//!
//! The Student-t observed information `(ν+1)(A−r²)/D²` is negative on outlying rows, so the
//! inner P-IRLS falls back to the Fisher (EM) weight and, after repeated fallbacks, runs the
//! Fisher fixed point with AA(1) acceleration. The accelerated candidate is an extrapolation of
//! the Levenberg–Marquardt step, not the step whose predicted reduction the gain ratio is taken
//! against. Its rejections were fed to the damping update as if the LM step had failed: every
//! iteration rejected the AA trial, Moré's interpolation hit its ×10 floor, the accepted plain
//! step divided the damping by only 3, and the damping ratcheted up ×3.3 per iteration. The inner
//! solve crawled, the outer ARC saw an objective the inner solve had not resolved, and the fit
//! refused with `trust_region_reject_floor` at `|Pg| = 3.1e1` after 67 outer iterations.
//!
//! A rejected AA trial now retries the plain LM step at the same damping. The fixture is the
//! sweep's seed-2 n = 100 `te` cell, exported from `bench/pygam_compare/worker.py::make_data`.
//!
//! The binary installs the production Laplace corrector, as `src/lib.rs` does for the wheel.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};

const TE_N100_SEED2: &str = include_str!("fixtures/student_t_te_n100_seed2.csv");

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

#[test]
fn student_t_tensor_smooth_certifies_through_anderson_rejections() {
    drop(
        gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
            gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
        )),
    );
    let data = dataset(TE_N100_SEED2);
    let config = FitConfig {
        family: Some("student-t".to_string()),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula("y ~ te(x0, x1)", &data, &config) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("a tensor-smooth Student-t formula must produce a standard fit"),
        Err(error) => panic!(
            "the Student-t tensor fit must certify its outer optimum: {error}"
        ),
    };
    assert!(
        fit.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
        "the certified smoothing parameters must be finite: {:?}",
        fit.fit.log_lambdas,
    );
}
