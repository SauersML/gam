//! Failing-ticket regression: a 1D smooth fit must succeed and produce
//! finite predictions even when the input dataset contains many duplicate
//! x values. This is common in real datasets (rounded measurements,
//! categorical proxies). The fit should treat duplicates as repeated
//! observations and compute the correct mean response.
//!
//! Setup: 200 rows where x ∈ {0.1, 0.3, 0.5, 0.7, 0.9} (5 unique values,
//! 40 reps each). Truth = sin(2π x). Fit `s(x, k=4)`. Predict at all 5
//! unique x values: average residual should match noise variance, and all
//! betas / predictions must be finite.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[test]
fn duplicate_x_rows_produces_finite_fit_and_reasonable_predictions() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(83);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let unique = [0.1, 0.3, 0.5, 0.7, 0.9];
    let reps = 40;
    let n = unique.len() * reps;

    let f = |t: f64| (2.0 * std::f64::consts::PI * t).sin();
    let mut headers: Vec<String> = vec!["x".into(), "y".into()];
    let _ = &mut headers; // keep allocator quiet
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for &xu in &unique {
        for _ in 0..reps {
            let y = f(xu) + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![xu.to_string(), y.to_string()]));
        }
    }
    let data = encode_recordswith_inferred_schema(
        ["x", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode duplicate-x dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=4)", &data, &cfg)
        .expect("fit with duplicate x rows should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "duplicate-x fit produced non-finite betas"
    );

    // Predict at unique x values; each prediction should be within 0.10 of
    // the truth (n=40 reps per point, σ=0.05 → SE of mean ≈ 0.008, so 0.10
    // is enormous slack).
    let mut m = Array2::<f64>::zeros((unique.len(), 2));
    for (i, &xu) in unique.iter().enumerate() {
        m[[i, 0]] = xu;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "duplicate-x predictions contained non-finite values"
    );

    let mut violations = Vec::<String>::new();
    for (i, &xu) in unique.iter().enumerate() {
        let want = f(xu);
        let got = pred[i];
        let err = (got - want).abs();
        eprintln!(
            "[dup-x] x={xu:.2} truth={want:+.3} pred={got:+.3} err={err:.3}"
        );
        if err > 0.10 {
            violations.push(format!(
                "x={xu:.2}: |pred {got:+.3} − truth {want:+.3}| = {err:.3} > 0.10"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "duplicate-x fit predictions diverge from per-cluster mean:\n  - {}",
        violations.join("\n  - "),
    );
}
