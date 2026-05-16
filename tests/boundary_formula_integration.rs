use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn boundary_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..28)
        .map(|i| {
            let x = i as f64 / 27.0;
            let y = 0.5 + 2.0 * x * x * (1.0 - x) * (1.0 - x);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode boundary dataset")
}

#[test]
fn fit_from_formula_accepts_bspline_endpoint_boundary_conditions() {
    init_parallelism();
    let data = boundary_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=0.5, bc_right=clamped, k=8)",
        &data,
        &config,
    )
    .expect("boundary-conditioned formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(!fit.design.smooth.terms[0].coeff_range.is_empty());
    let constraints = fit
        .design
        .linear_constraints
        .as_ref()
        .expect("nonzero anchor should be enforced as linear constraints");
    let lhs = constraints.a.dot(&fit.fit.beta);
    for i in 0..lhs.len() {
        assert!(
            lhs[i] + 1e-6 >= constraints.b[i],
            "boundary constraint {i} violated: lhs={} rhs={}",
            lhs[i],
            constraints.b[i]
        );
    }
    assert!(
        constraints.b.iter().any(|v| (*v - 0.5).abs() < 1e-12),
        "nonzero anchor target was not carried into constraints: {:?}",
        constraints.b
    );
}
