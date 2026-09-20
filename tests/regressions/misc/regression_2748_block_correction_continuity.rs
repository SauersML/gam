//! #2748: the outer criterion must be a FUNCTION of ρ, and the `haberman_5yr`
//! model must fit.
//!
//! The `#784` block-local Gauss–Hermite correction used to be added to the
//! REML/LAML criterion only where a predicate evaluated at ρ held — the
//! skewness threshold `τ(n_eff) = sqrt((24/5)/n_eff)`, the paired-rule
//! resolution test, and the block-dimension cap. A term that a ρ-predicate
//! switches gives the criterion a jump of the full `|Δ_b|`, and the outer
//! search descends ONTO that surface, because declining the correction raises
//! the cost by `Δ_b` and so the ON region's minimum sits on its boundary.
//!
//! Measured on this fixture before the repair: `max|γ|` walked
//! `0.141 → 0.126 → 0.125` onto `τ = 0.125` and stopped, `V` jumped
//! `1.744282e2 → 1.744593e2` — exactly `Δ_b = 3.1144e-2` — between two adjacent
//! line-search trial points at `|g| = 2.045e-2`, and the fit died with
//! `line_search=StepSizeTooSmall after 50 attempt(s)` after 112.9 s.
//!
//! Two tests, from two directions:
//!
//! 1. [`haberman_double_penalty_binomial_fit_converges_2748`] — the symptom.
//!    The model the benchmark cell fits must fit.
//! 2. [`outer_criterion_has_no_jump_across_the_block_correction_threshold_2748`]
//!    — the cause, and the one that would catch a regression from a different
//!    angle. It sweeps ρ along a segment that crosses the `τ` surface and
//!    checks the criterion's SECOND difference against its first: a smooth
//!    function's second difference falls with the step, a jump's does not.
//!    This test fails on the pre-repair code even if the fit above were to
//!    converge by luck of the seed.

use csv::StringRecord;
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost, evaluate_externalgradient,
};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array1;

const HABERMAN_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/haberman.csv");
const FEATURES: [&str; 3] = ["age", "op_year", "axil_nodes"];

/// `bench/datasets/haberman.csv`, the first four columns, `status == 2` as the
/// positive class, then each feature centred and scaled — the benchmark cell's
/// own preparation (`_load_haberman_dataset` + `zscore_train_test`).
fn haberman_z_scored() -> (Vec<[f64; 3]>, Vec<f64>) {
    let text = std::fs::read_to_string(HABERMAN_CSV).expect("bench/datasets/haberman.csv");
    let mut features: Vec<[f64; 3]> = Vec::new();
    let mut response: Vec<f64> = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let Some(values) = fields[..4]
            .iter()
            .map(|f| f.trim().parse::<f64>().ok())
            .collect::<Option<Vec<f64>>>()
        else {
            continue;
        };
        features.push([values[0], values[1], values[2]]);
        response.push(if values[3].round() as i64 == 2 { 1.0 } else { 0.0 });
    }
    assert!(
        features.len() > 300,
        "haberman.csv must carry its ~306 rows; got {}",
        features.len()
    );
    let n = features.len() as f64;
    for j in 0..3 {
        let mean = features.iter().map(|row| row[j]).sum::<f64>() / n;
        let variance = features.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / n;
        let scale = if variance > 0.0 { variance.sqrt() } else { 1.0 };
        for row in features.iter_mut() {
            row[j] = (row[j] - mean) / scale;
        }
    }
    (features, response)
}

fn haberman_dataset() -> gam::data::EncodedDataset {
    let (features, response) = haberman_z_scored();
    let headers: Vec<String> = FEATURES
        .iter()
        .map(|name| (*name).to_string())
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
    encode_recordswith_inferred_schema(headers, records).expect("encode haberman")
}

fn formula_for(features: &[&str]) -> String {
    let body = features
        .iter()
        .map(|name| format!("s({name}, bs=ps, knots=8, double_penalty=true)"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!("y ~ {body}")
}

fn binomial_logit_config() -> FitConfig {
    FitConfig {
        family: Some("binomial-logit".to_string()),
        ..FitConfig::default()
    }
}

/// The benchmark cell itself: three double-penalised P-splines, binomial-logit,
/// 306 rows. Before the repair this ran 151.2 s and refused
/// (`NOT STATIONARY (|Pg|=1.101e0 > bound=3.636e-6)`, `StepSizeTooSmall after
/// 50 attempt(s)`); the two-smooth sub-model refused after 112.9 s.
///
/// Both are asserted, because the two-smooth one is the minimal cell and a
/// repair that only rescued the larger model would be rescuing it by accident.
#[test]
fn haberman_double_penalty_binomial_fit_converges_2748() {
    init_parallelism();
    let data = haberman_dataset();
    let config = binomial_logit_config();
    for features in [&["age", "axil_nodes"][..], &FEATURES[..]] {
        let formula = formula_for(features);
        let outcome = gam::fit_from_formula(&formula, &data, &config);
        let fit = outcome.unwrap_or_else(|error| {
            panic!(
                "#2748: `{formula}` must fit. The outer search cannot converge on a criterion \
                 that a rho-predicate switches a 3e-2 term in and out of: {error}"
            )
        });
        let FitResultShape::Standard(reml_score) = fit_shape(&fit) else {
            panic!("the haberman cell is a standard GLM fit");
        };
        assert!(
            reml_score.is_finite(),
            "#2748: `{formula}` minted a fit with no finite REML/LAML criterion"
        );
    }
}

enum FitResultShape {
    Standard(f64),
    Other,
}

fn fit_shape(fit: &gam::FitResult) -> FitResultShape {
    match fit {
        gam::FitResult::Standard(standard) => {
            FitResultShape::Standard(standard.fit.reml_score().unwrap_or(f64::NAN))
        }
        _ => FitResultShape::Other,
    }
}

/// The cause, tested directly: `V(ρ)` has no jump.
///
/// The sweep is along `−∇V` from the point the pre-repair fit refused at, which
/// is where the `τ` surface is (the descent converges onto it). A smooth `V`
/// sampled on a uniform ladder of spacing `h` has first differences that vary
/// by `O(h²)` between adjacent intervals; a jump of `J` puts `J` into ONE
/// second difference no matter how small `h` is. So the discriminator is the
/// max absolute second difference measured against the step — and the
/// pre-repair criterion fails it by three orders at `h = 1e-3`.
#[test]
fn outer_criterion_has_no_jump_across_the_block_correction_threshold_2748() {
    init_parallelism();
    let data = haberman_dataset();
    let config = binomial_logit_config();
    let formula = formula_for(&["age", "axil_nodes"]);

    let model = gam::materialize(&formula, &data, &config).expect("materialize haberman");
    let FitRequest::Standard(request) = model.request else {
        panic!("the haberman cell is a standard GLM request");
    };
    let design = build_term_collection_design(request.data.view(), &request.spec)
        .expect("build the haberman term-collection design");
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
        persistent_warm_start_store: None,
    };
    let rho_dim = design.penalties.len();
    assert_eq!(
        rho_dim, 4,
        "two double-penalised smooths carry four rho coordinates"
    );

    // The checkpoint the pre-repair refusal published. It is where `max|γ|` had
    // walked onto `τ`, i.e. exactly where a jump would be if there is one.
    let anchor = Array1::from(vec![
        6.507_261_272_219_448,
        -2.211_492_351_076_335_3,
        4.996_412_273_654_684,
        -1.046_274_404_504_386_8,
    ]);
    let value_at = |theta: &Array1<f64>| -> f64 {
        evaluate_externalcost(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &design.penalties,
            &opts,
            theta,
        )
        .expect("outer criterion value")
    };
    let gradient = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &design.penalties,
        &opts,
        &anchor,
    )
    .expect("analytic outer gradient");
    let gradient_norm = gradient.dot(&gradient).sqrt();
    assert!(
        gradient_norm.is_finite() && gradient_norm > 0.0,
        "#2748: the sweep needs a direction; |g| = {gradient_norm}"
    );
    let direction = &gradient / gradient_norm;

    // A ladder centred on the anchor so the crossing is interior to it.
    const STEPS: usize = 81;
    const STEP: f64 = 1.0e-3;
    let values: Vec<f64> = (0..STEPS)
        .map(|i| {
            let alpha = (i as f64 - (STEPS as f64 - 1.0) / 2.0) * STEP;
            value_at(&(&anchor - &(&direction * alpha)))
        })
        .collect();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "#2748: the criterion must be finite along the whole sweep"
    );

    let first: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
    let worst_second = first
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);
    let typical_first = first.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

    // `V` is smooth, so `|ΔΔV| = |V''|·h² + O(h³)`; with `h = 1e-3` and the
    // criterion's curvature `O(1)` that is `~1e-6`, while one first difference
    // is `|g|·h ~ 3e-4`. A jump of the correction's own size (`3.1e-2`,
    // measured) lands whole in one second difference — 100x the first
    // difference rather than 1/100 of it. The bar is the geometric middle of
    // those two regimes, so it separates them by two orders on either side and
    // is not a tolerance fitted to the fix.
    assert!(
        worst_second <= typical_first,
        "#2748: the outer criterion has a JUMP along rho. Worst second difference \
         {worst_second:.6e} exceeds the largest first difference {typical_first:.6e} at step \
         {STEP:.0e}; a smooth criterion's second difference is O(h^2) and its first is O(h), so \
         this is a discontinuity, not curvature. That is what a rho-predicate switching the #784 \
         block-local correction in and out of the criterion does, and no line search can cross it."
    );
}
