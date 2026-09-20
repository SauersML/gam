//! #3951 profile of the `te(x0, x1)` head-to-head cell of bench/pygam_compare
//! (n = 1000, X ~ U(0,1)^2, eta = sin(2 pi x0) cos(2 pi x1); Gaussian sd 0.5,
//! binomial logit 1.5 eta, Poisson log 0.5 + 0.7 eta).
//!
//! Prints, per family and seed, the basis width, the outer work the fit did
//! (`outer_iterations` of the certified solve, `outer_cost_evals`,
//! `inner_pirls_solves`), the wall time of the fit and of rebuilding the
//! design from the frozen spec, and the link-scale RMSE against the truth on a
//! held-out draw. One seed per family is then refitted with the solver's debug
//! trace on, whose elapsed-time stamps show where the fit's time goes.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use std::time::Instant;

const N: usize = 1000;

fn truth(x0: f64, x1: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x0).sin() * (2.0 * std::f64::consts::PI * x1).cos()
}

/// The family's linear predictor as a function of the bench truth `eta`.
fn link_truth(family: &str, eta: f64) -> f64 {
    match family {
        "gaussian" => eta,
        "binomial" => 1.5 * eta,
        "poisson" => 0.5 + 0.7 * eta,
        other => panic!("no #3951 generator for {other}"),
    }
}

fn draw(family: &str, seed: u64) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.5).expect("normal");
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let p = [u.sample(&mut rng), u.sample(&mut rng)];
        let lin = link_truth(family, truth(p[0], p[1]));
        let yi = match family {
            "gaussian" => lin + noise.sample(&mut rng),
            "binomial" => {
                let mu = 1.0 / (1.0 + (-lin).exp());
                f64::from(u.sample(&mut rng) < mu)
            }
            _ => Poisson::new(lin.exp()).expect("poisson mean").sample(&mut rng),
        };
        x.push(p);
        y.push(yi);
    }
    (x, y)
}

struct Profile {
    fit: StandardFitResult,
    fit_secs: f64,
    design_secs: f64,
    rmse_link: f64,
}

fn fit_once(family: &str, seed: u64) -> Profile {
    let (x, y) = draw(family, seed);
    let headers = ["x0", "x1", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![x[i][0].to_string(), x[i][1].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode te dataset");
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let t0 = Instant::now();
    let result = fit_from_formula("y ~ te(x0, x1)", &ds, &cfg).expect("te fit");
    let fit_secs = t0.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for te(x0, x1)");
    };

    let (xt, _) = draw(family, seed + 1000);
    let col = ds.column_map();
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for (i, p) in xt.iter().enumerate() {
        grid[[i, col["x0"]]] = p[0];
        grid[[i, col["x1"]]] = p[1];
    }
    let t1 = Instant::now();
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design");
    let design_secs = t1.elapsed().as_secs_f64();
    let eta = design.design.apply(&fit.fit.beta);
    let sse: f64 = xt
        .iter()
        .zip(eta.iter())
        .map(|(p, e)| (e - link_truth(family, truth(p[0], p[1]))).powi(2))
        .sum();
    Profile {
        fit,
        fit_secs,
        design_secs,
        rmse_link: (sse / N as f64).sqrt(),
    }
}

fn report(family: &str, seed: u64, p: &Profile) {
    let f = &p.fit.fit;
    eprintln!(
        "RECORD_3951 family={family} seed={seed} p={} lambdas={} edf={:.2} \
         outer_iterations={} outer_cost_evals={} inner_pirls_solves={} inner_cycles={} \
         fit_s={:.4} design_rebuild_s={:.4} rmse_link={:.5}",
        f.beta.len(),
        f.lambdas.len(),
        f.edf_total().unwrap_or(f64::NAN),
        f.outer_iterations,
        f.outer_cost_evals,
        f.inner_pirls_solves,
        f.inner_cycles,
        p.fit_secs,
        p.design_secs,
        p.rmse_link,
    );
}

#[test]
fn te_head_to_head_profile_3951() {
    init_parallelism();
    for family in ["gaussian", "binomial", "poisson"] {
        for seed in 0..3u64 {
            let profile = fit_once(family, seed);
            report(family, seed, &profile);
            assert!(
                profile.rmse_link.is_finite() && profile.fit.fit.beta.iter().all(|b| b.is_finite()),
                "{family} seed {seed}: te fit must be finite"
            );
        }
    }
    // Solver trace for one seed per family: the elapsed stamps on each line
    // locate the fit's time between basis construction, seeding, the outer
    // search and the post-fit inference.
    gam::progress_log::init_logging_at(log::LevelFilter::Debug);
    for family in ["gaussian", "binomial", "poisson"] {
        eprintln!("TRACE_3951_BEGIN family={family}");
        let profile = fit_once(family, 0);
        report(family, 0, &profile);
        eprintln!("TRACE_3951_END family={family}");
    }
    log::set_max_level(log::LevelFilter::Warn);
}
