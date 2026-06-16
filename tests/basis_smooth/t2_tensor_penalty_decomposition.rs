//! Regression for issue #1185: `t2(...)` must not be silently aliased to
//! `te(...)`.  `te(x, z)` emits one overlapping Kronecker-sum penalty per
//! margin.  `t2(x, z)` emits mgcv-style separable penalties for the marginal
//! penalized/null tensor subspaces.

use csv::StringRecord;
use gam::basis::PenaltySource;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

fn grid_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    let grid = 18usize;
    for i in 0..grid {
        for j in 0..grid {
            let x = i as f64 / (grid as f64 - 1.0);
            let z = j as f64 / (grid as f64 - 1.0);
            let y = (2.0 * x).sin() + (3.0 * z).cos() + x * z;
            rows.push(StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode t2 grid dataset")
}

fn fit(formula: &str) -> StandardFitResult {
    init_parallelism();
    let data = grid_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &config)
        .unwrap_or_else(|err| panic!("{formula} should fit: {err:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for {formula}");
    };
    fit
}

#[test]
fn t2_uses_separable_penalty_decomposition_not_te_marginal_alias() {
    let te = fit("y ~ te(x, z, k=5)");
    let t2 = fit("y ~ t2(x, z, k=5)");

    let te_term = &te.design.smooth.terms[0];
    let t2_term = &t2.design.smooth.terms[0];

    assert_eq!(
        te_term.penalties_local.len(),
        2,
        "te has one penalty per margin"
    );
    assert_eq!(
        t2_term.penalties_local.len(),
        3,
        "2D t2 with one marginal penalty per axis has range-null, null-range, and range-range components"
    );
    assert_ne!(
        te_term.penalties_local.len(),
        t2_term.penalties_local.len(),
        "t2 must not be a te penalty alias"
    );

    assert!(
        te_term
            .penaltyinfo_local
            .iter()
            .all(|info| matches!(info.source, PenaltySource::TensorMarginal { .. })),
        "te penalties should retain marginal Kronecker-sum sources"
    );

    let mut separable_components = t2_term
        .penaltyinfo_local
        .iter()
        .map(|info| match &info.source {
            PenaltySource::TensorSeparable { penalized_margins } => penalized_margins.clone(),
            other => panic!("unexpected t2 penalty source: {other:?}"),
        })
        .collect::<Vec<_>>();
    separable_components.sort();
    assert_eq!(
        separable_components,
        vec![vec![0], vec![0, 1], vec![1]],
        "t2 should split the coefficient space by marginal range/null tensor subspaces"
    );

    let t2_width = t2_term.coeff_range.len();
    assert!(
        t2_term
            .penalties_local
            .iter()
            .all(|s| s.nrows() == t2_width && s.ncols() == t2_width),
        "every active t2 penalty must live in the transformed term coefficient space"
    );
}
