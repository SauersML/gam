//! #2748 / `haberman_5yr`: is the analytic ρ-gradient the gradient of the
//! criterion the line search evaluates?
//!
//! The cell dies with `line_search=StepSizeTooSmall after 50 attempt(s)` — "the
//! direction descended but no step improved the objective". That has exactly
//! two causes, and this probe separates them by measuring both sides at one ρ:
//!
//! * the analytic gradient `∇V(ρ)` the outer search steers by, and
//! * a central finite difference of `V(ρ)` — the same criterion, through the
//!   same public evaluator — at a ladder of steps.
//!
//! It also walks the criterion ALONG the analytic descent direction over
//! fifteen decades of step length, which is the line search's own question:
//! if `V(ρ − α·g)` never falls below `V(ρ)` for any `α`, the objective is not
//! the gradient's antiderivative and no globalization can rescue it.
//!
//! Run:
//!   cargo run --release --example probe_2748_haberman_outer_fd -- [features] [rho...]
//! e.g.
//!   ... -- age,axil_nodes 6.507261272219448 -2.2114923510763353 \
//!        4.996412273654684 -1.0462744045043868
//! With no ρ given the probe uses the ρ the fit's own refusal published as its
//! checkpoint for that feature set.
//!
//! NOT a test — examples skip dev-deps, so the CSV read and the z-scoring are
//! inlined exactly as in `probe_2748_haberman_outer`.

use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism};
use csv::StringRecord;
use ndarray::Array1;

const FEATURES: [&str; 3] = ["age", "op_year", "axil_nodes"];

fn load_rows() -> (Vec<[f64; 3]>, Vec<f64>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/datasets/haberman.csv");
    let text = std::fs::read_to_string(&path).expect("bench/datasets/haberman.csv");
    let mut features = Vec::new();
    let mut response = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let parsed: Option<Vec<f64>> = fields[..4].iter().map(|f| f.trim().parse().ok()).collect();
        let Some(values) = parsed else { continue };
        features.push([values[0], values[1], values[2]]);
        response.push(if values[3].round() as i64 == 2 { 1.0 } else { 0.0 });
    }
    (features, response)
}

fn zscore(features: &mut [[f64; 3]]) {
    let n = features.len() as f64;
    for j in 0..3 {
        let mean = features.iter().map(|row| row[j]).sum::<f64>() / n;
        let variance = features.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / n;
        let sd = variance.sqrt();
        let scale = if sd > 0.0 { sd } else { 1.0 };
        for row in features.iter_mut() {
            row[j] = (row[j] - mean) / scale;
        }
    }
}

fn main() {
    init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    let selected: Vec<&str> = match args.get(1) {
        None => vec!["age", "axil_nodes"],
        Some(raw) => raw
            .split(',')
            .map(|name| {
                *FEATURES
                    .iter()
                    .find(|feature| **feature == name.trim())
                    .unwrap_or_else(|| panic!("unknown haberman feature '{name}'"))
            })
            .collect(),
    };
    let requested_rho: Vec<f64> = args[2.min(args.len())..]
        .iter()
        .filter_map(|a| a.parse::<f64>().ok())
        .collect();

    let (mut features, response) = load_rows();
    zscore(&mut features);
    let headers: Vec<String> = FEATURES
        .iter()
        .map(|name| name.to_string())
        .chain(std::iter::once("y".to_string()))
        .collect();
    let records: Vec<StringRecord> = features
        .iter()
        .zip(response.iter())
        .map(|(row, y)| {
            StringRecord::from(vec![
                row[0].to_string(),
                row[1].to_string(),
                row[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, records).expect("encode");

    let body = selected
        .iter()
        .map(|name| format!("s({name}, type=ps, knots=8, double_penalty=true)"))
        .collect::<Vec<_>>()
        .join(" + ");
    let formula = format!("y ~ {body}");
    let config = FitConfig {
        family: Some("binomial-logit".to_string()),
        ..FitConfig::default()
    };

    // The FIT path's own geometry: the same spec, design, penalties and
    // null-space dimensions the outer search steers over. Rebuilding a P-spline
    // basis by hand here would audit a different model than the one that fails.
    let model = gam::materialize(&formula, &data, &config).expect("materialize");
    let FitRequest::Standard(request) = model.request else {
        panic!("the haberman cell is a standard GLM request");
    };
    let design = build_term_collection_design(request.data.view(), &request.spec)
        .expect("build the term-collection design");
    let x = design.design.clone();
    let y: Array1<f64> = (*request.y).clone();
    let weights: Array1<f64> = (*request.weights).clone();
    let offset: Array1<f64> = &*request.offset + &design.affine_offset;

    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };

    let rho_dim = design.penalties.len();
    println!(
        "[2748-fd] formula: {formula}\n[2748-fd] n={} p={} rho_dim={rho_dim} nullspace_dims={:?}",
        x.nrows(),
        x.ncols(),
        design.nullspace_dims,
    );

    // The checkpoints the fit's own refusal published, so the audit lands where
    // the line search gave up rather than at a cold seed.
    let default_rho: Vec<f64> = match selected.as_slice() {
        ["age", "axil_nodes"] => vec![
            6.507_261_272_219_448,
            -2.211_492_351_076_335_3,
            4.996_412_273_654_684,
            -1.046_274_404_504_386_8,
        ],
        _ => vec![0.0; rho_dim],
    };
    let rho = Array1::from(if requested_rho.len() == rho_dim {
        requested_rho
    } else {
        assert_eq!(
            default_rho.len(),
            rho_dim,
            "no published checkpoint for {selected:?}; pass {rho_dim} rho values"
        );
        default_rho
    });
    println!("[2748-fd] rho = {:?}", rho.to_vec());

    let value_at = |theta: &Array1<f64>| -> f64 {
        evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &design.penalties,
            &opts,
            theta,
        )
        .expect("outer criterion value")
        .0
    };

    let analytic = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &design.penalties,
        &opts,
        &rho,
    )
    .expect("analytic outer gradient");
    let base = value_at(&rho);
    println!(
        "[2748-fd] V(rho) = {base:.12e}   |analytic| = {:.6e}",
        analytic.dot(&analytic).sqrt()
    );

    // ── 1. Reproducibility of the criterion at ONE point ─────────────────────
    //
    // Every conclusion below is a difference of two evaluations, so the floor
    // those differences can resolve has to be measured first — and it is not
    // assumed to be machine epsilon, because a warm-started inner solve makes
    // the criterion a function of the PATH as well as the point.
    let repeats: Vec<f64> = (0..5).map(|_| value_at(&rho)).collect();
    let spread = repeats
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| a.max(*b))
        - repeats.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    println!("[2748-fd] value reproducibility over 5 re-evaluations: spread = {spread:.6e}");

    // ── 2. Central differences, per coordinate, over a step ladder ───────────
    println!("[2748-fd] --- per-coordinate central differences ---");
    for k in 0..rho_dim {
        let mut row = format!("[2748-fd] d/drho_{k}: analytic = {:+.9e}", analytic[k]);
        for &step in &[1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6] {
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus[k] += step;
            minus[k] -= step;
            let fd = (value_at(&plus) - value_at(&minus)) / (2.0 * step);
            row.push_str(&format!("  fd({step:.0e}) = {fd:+.9e}"));
        }
        println!("{row}");
    }

    // ── 3. The line search's own question, along the analytic direction ──────
    //
    // Steepest descent, not the BFGS direction: a wrong metric can only rotate
    // the step, and `−g` is a descent direction for ANY objective whose
    // gradient this is. If `V(rho − alpha*g)` does not fall below `V(rho)` for
    // any alpha over fifteen decades, `g` is not that objective's gradient.
    let grad_norm = analytic.dot(&analytic).sqrt();
    println!(
        "[2748-fd] --- V(rho - alpha*g/|g|) - V(rho), steepest descent, |g| = {grad_norm:.6e} ---"
    );
    let direction = &analytic / grad_norm;
    for exponent in 0..16 {
        let alpha = 10f64.powi(-(exponent as i32));
        let trial = &rho - &(&direction * alpha);
        let value = value_at(&trial);
        println!(
            "[2748-fd] alpha = {alpha:.0e}   dV = {:+.9e}   predicted = {:+.9e}",
            value - base,
            -alpha * grad_norm
        );
    }
}
