//! #2668 / #1266 diagnostic: is gam's shipped REML optimum on the irrelevant-covariate
//! seeds the minimum of gam's OWN criterion, or does the outer search stop short?
//!
//! On seeds 201-203 gam's `s(z)` EDF (2.76 / 1.35 / 1.93) exceeds mgcv's REML optimum on
//! the same cubic B-spline basis with a second-derivative penalty (1.84 / 1.01 / 1.76). Cross-tool
//! criterion values carry a constant offset, so this probe stays inside gam: it re-evaluates
//! the production external REML cost on the fitted design at the fitted rho and along each
//! outer coordinate. A coordinate move that lowers the cost by more than the criterion band
//! means the shipped fit is not a local minimum of its own criterion.
//!
//! Positive control: the cost at the fitted rho must reproduce the fit's reported
//! `reml_score()`; if it does not, the probe is evaluating a different criterion and every
//! profile row is void.
//!
//! Data generation is byte-for-byte the #1266 contract's `irrelevant_covariate_dataset`.
//! Compile against an idle lane's existing gam library graph (no dependency rebuild):
//!   python scripts/compile_warm_probe.py bench/measurements/issue_2668/reml_profile_1266_probe.rs \
//!     <out-bin> --target-dir <lane-target> --anchor gam-<hash> \
//!     --extern ndarray --extern rand --extern rand_distr --extern csv --extern gam_solve --test
//! then run `<out-bin> --test-threads=1 --nocapture` with RAYON_NUM_THREADS=2.

use csv::StringRecord;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use gam_solve::estimate::{ExternalOptimOptions, evaluate_externalcost_andridge, smooth_term_summary_rows};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn irrelevant_covariate_dataset(seed: u64, n: usize) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let mut response = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let y = (6.0_f64 * x).sin() + noise.sample(&mut rng);
            response.push(y);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode");
    (data, response)
}

#[test]
fn reml_profile_1266_probe() {
    init_parallelism();
    let cfg = FitConfig::default();
    for seed in 200u64..205 {
        let (data, response) = irrelevant_covariate_dataset(seed, 800);
        let fit = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("full fit");
        let FitResult::Standard(std_fit) = &fit else {
            panic!("expected a standard Gaussian fit");
        };
        let rows = smooth_term_summary_rows(&std_fit.design, &std_fit.resolvedspec, &std_fit.fit, None);
        let edfs: Vec<(String, f64)> = rows.iter().map(|row| (row.name.clone(), row.edf)).collect();
        let n = response.len();
        let y = Array1::from(response);
        let w = Array1::<f64>::ones(n);
        let offset = std_fit
            .design
            .compose_offset(Array1::<f64>::zeros(n).view(), "1266 probe")
            .expect("compose offset");
        let opts = ExternalOptimOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            family: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            compute_inference: true,
            skip_rho_posterior_inference: true,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: std_fit.design.nullspace_dims.clone(),
            linear_constraints: std_fit.design.linear_constraints.clone(),
            firth_bias_reduction: Some(false),
            rho_prior: Default::default(),
            kronecker_penalty_system: std_fit.design.kronecker_penalty_system(),
            kronecker_factored: None,
            persistent_warm_start_store: None,
        };
        let rho_hat = std_fit.fit.log_lambdas.clone();
        let cost = |rho: &Array1<f64>| {
            evaluate_externalcost_andridge(
                y.view(),
                w.view(),
                std_fit.design.design.clone(),
                offset.view(),
                &std_fit.design.penalties,
                &opts,
                rho,
            )
            .map(|(value, _)| value)
        };
        let base = cost(&rho_hat).expect("cost at the fitted rho");
        let reported = std_fit.fit.reml_score();
        let gradient = std_fit.fit.outer_gradient_norm;
        let band = f64::EPSILON.sqrt() * (1.0 + base.abs());
        eprintln!(
            "[1266-probe seed={seed}] edf={edfs:?} rho_hat={:?} cost(rho_hat)={base:.12e} reml_score={reported:?} \
             control_gap={:?} terminal_grad={gradient:?} band={band:.3e}",
            rho_hat.to_vec(),
            reported.map(|value| base - value),
        );
        for coordinate in 0..rho_hat.len() {
            let mut line = String::new();
            for delta in [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0, 8.0, 20.0] {
                let mut moved = rho_hat.clone();
                moved[coordinate] += delta;
                match cost(&moved) {
                    Ok(value) => line.push_str(&format!(" d{delta:+}:{:+.4e}", value - base)),
                    Err(error) => line.push_str(&format!(" d{delta:+}:ERR({error})")),
                }
            }
            eprintln!("[1266-probe seed={seed}] coordinate {coordinate} cost-minus-base:{line}");
        }
    }
}
