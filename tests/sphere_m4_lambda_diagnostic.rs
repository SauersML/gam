//! Diagnostic: dump REML-chosen lambdas / log_lambdas / beta-norm / pred-variance
//! for sphere(lat, lon, k=30, m=1..4). Hypothesis under investigation: at m=4
//! REML picks λ→∞ and the smooth contribution collapses to a near-constant
//! (the response mean), while m=1, 2, 3 fit the truth fine.
//!
//! This is a *diagnostic* — it never asserts pass/fail, it just logs numbers
//! so we can confirm or refute the lambda-collapse hypothesis.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn sphere_m_sweep_lambda_diagnostic() {
    assert!(file!().ends_with(".rs"));
    init_parallelism();
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // Compute response mean for reference (where collapsed fits should land).
    let n_train = 400usize;
    eprintln!("\n=========================================================");
    eprintln!("[diag] sphere m-sweep, n=400, seed=7, truth peak-to-peak ~1.4");
    eprintln!("=========================================================\n");

    for m in [1usize, 2, 3, 4] {
        let formula = format!("y ~ sphere(lat, lon, k=30, m={m})");
        eprintln!("---- m={m} : `{formula}` ----");
        let result = match fit_from_formula(&formula, &data, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[diag] m={m} FIT ERROR: {e}");
                continue;
            }
        };
        let FitResult::Standard(fit) = result else {
            eprintln!("[diag] m={m} non-standard result");
            continue;
        };

        // ── REML / smoothing summary ────────────────────────────────────
        let lambdas = &fit.fit.lambdas;
        let log_lambdas = &fit.fit.log_lambdas;
        let beta = &fit.fit.beta;
        let beta_l2 = beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let outer_iters = fit.fit.outer_iterations;
        let outer_converged = fit.fit.outer_converged;
        let reml_score = fit.fit.reml_score;
        let outer_grad_norm = fit.fit.outer_gradient_norm;

        eprintln!("[diag] m={m} lambdas        = {:?}", lambdas.to_vec());
        eprintln!("[diag] m={m} log_lambdas    = {:?}", log_lambdas.to_vec());
        eprintln!(
            "[diag] m={m} beta.len={}  beta_L2={:.6e}",
            beta.len(),
            beta_l2
        );
        eprintln!(
            "[diag] m={m} REML score     = {reml_score:.6}   outer_iters={outer_iters}  \
             outer_converged={outer_converged}  outer_grad_norm={outer_grad_norm:?}"
        );

        // ── 15×15 grid prediction variance / range ──────────────────────
        let mut pts = Vec::new();
        for i in 0..15 {
            let lat = -75.0 + 150.0 * (i as f64) / 14.0;
            for j in 0..15 {
                let lon = -175.0 + 350.0 * (j as f64) / 14.0;
                pts.push((lat, lon));
            }
        }
        let n = pts.len();
        let mut design_input = Array2::<f64>::zeros((n, 3));
        for (i, (lat, lon)) in pts.iter().enumerate() {
            design_input[[i, 0]] = *lat;
            design_input[[i, 1]] = *lon;
        }
        let design =
            build_term_collection_design(design_input.view(), &fit.resolvedspec).expect("design");
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        let mean_pred = pred.iter().sum::<f64>() / pred.len() as f64;
        let var_pred =
            pred.iter().map(|v| (v - mean_pred).powi(2)).sum::<f64>() / pred.len() as f64;
        let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let truth: Vec<f64> = pts
            .iter()
            .map(|(lat, lon)| {
                0.5 + 0.6 * lat.to_radians().sin()
                    + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            })
            .collect();
        let truth_mean = truth.iter().sum::<f64>() / truth.len() as f64;
        let truth_var =
            truth.iter().map(|v| (v - truth_mean).powi(2)).sum::<f64>() / truth.len() as f64;
        let rmse = (pred
            .iter()
            .zip(truth.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / pred.len() as f64)
            .sqrt();

        eprintln!(
            "[diag] m={m} pred mean={mean_pred:.4}  pred var={var_pred:.4e}  \
             pred range=[{mn:.4}, {mx:.4}]"
        );
        eprintln!(
            "[diag] m={m} truth mean={truth_mean:.4} truth var={truth_var:.4e}  \
             rmse(pred vs truth)={rmse:.4}"
        );
        let collapsed = var_pred < 0.01 * truth_var;
        eprintln!(
            "[diag] m={m} COLLAPSED? {} (pred_var / truth_var = {:.3e})",
            if collapsed { "YES" } else { "no" },
            var_pred / truth_var
        );
        eprintln!();
        let _ = n_train; // silence unused if cfg shifts
    }
    eprintln!("=========================================================\n");
}
