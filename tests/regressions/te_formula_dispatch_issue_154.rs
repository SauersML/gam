//! Regression for GitHub issue #154: `te(x1, x2)` from the formula DSL must be
//! accepted as a tensor-product smooth, mirroring the example in the public
//! `fit` docstring (`"y ~ s(x1) + te(x2, x3)"`). Before the fix, this raised
//! `unsupported smooth type 'tensor'` during formula materialization because
//! the parser produced `SmoothKind::Te` (canonicalized to type="tensor") with
//! no matching arm in the smooth-basis builder.

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

#[test]
fn te_x1_x2_formula_fits_separable_truth() {
    init_parallelism();

    let n: usize = 300;
    let mut rng = StdRng::seed_from_u64(0);
    let ux = Uniform::new(-3.0_f64, 3.0_f64).expect("uniform domain");
    let noise = Normal::new(0.0_f64, 0.2_f64).expect("noise");

    let headers = ["x1", "x2", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut train_xs: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut sum_sq_signal = 0.0_f64;
    for _ in 0..n {
        let x1 = ux.sample(&mut rng);
        let x2 = ux.sample(&mut rng);
        let f = x1.sin() * x2.cos();
        let y = f + noise.sample(&mut rng);
        sum_sq_signal += f * f;
        train_xs.push((x1, x2));
        rows.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            y.to_string(),
        ]));
    }
    assert!(
        sum_sq_signal > 1.0,
        "sanity: separable truth has non-trivial signal"
    );

    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // The headline mgcv-style tensor-product smooth. With no explicit `type=`
    // it must canonicalize to a tensor-product basis (no longer surfaces
    // `unsupported smooth type 'tensor'`).
    let result =
        fit_from_formula("y ~ te(x1, x2)", &data, &cfg).expect("te(x1, x2) formula must fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // Rebuild the design at the training points and predict, then check
    // residual RMSE is well below the unit signal amplitude: the smooth
    // must actually fit the separable surface, not collapse to a constant.
    let mut design_in = Array2::<f64>::zeros((n, 3));
    for (i, &(x1, x2)) in train_xs.iter().enumerate() {
        design_in[[i, 0]] = x1;
        design_in[[i, 1]] = x2;
        design_in[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(design_in.view(), &fit.resolvedspec).expect("rebuild design");
    let yhat = design.design.apply(&fit.fit.beta);
    let mut sse = 0.0_f64;
    for (i, &(x1, x2)) in train_xs.iter().enumerate() {
        let f_true = x1.sin() * x2.cos();
        let r = yhat[i] - f_true;
        sse += r * r;
    }
    let rmse = (sse / n as f64).sqrt();
    assert!(
        rmse < 0.30,
        "te(x1, x2) tensor smooth fit too poorly: rmse={rmse:.4}"
    );
}
