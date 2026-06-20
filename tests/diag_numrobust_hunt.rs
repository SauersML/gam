// Numerical-robustness hunt: extreme data scale, degenerate inputs, zero-variance.
// DIAGNOSTIC — prints fit outcomes; final intentional panic to surface stdout.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn dataset_xy(xs: Vec<f64>, ys: Vec<f64>) -> gam::data::EncodedDataset {
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| StringRecord::from(vec![x.to_string(), y.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(["x", "y"].into_iter().map(String::from).collect(), rows)
        .expect("encode")
}

fn summarize(label: &str, fit: &Result<FitResult, gam::WorkflowError>) {
    match fit {
        Ok(FitResult::Standard(s)) => {
            let u = &s.fit;
            let any_nan = u.beta.iter().any(|b| !b.is_finite());
            let reml_bad = !u.reml_score.is_finite();
            let edf = u.inference.as_ref().map(|i| i.edf_total);
            println!(
                "{label}: OK converged={} reml={:.4} reml_finite={} beta_has_nonfinite={} edf={:?} loglam={:?}",
                u.outer_converged,
                u.reml_score,
                !reml_bad,
                any_nan,
                edf,
                u.log_lambdas.to_vec(),
            );
            if any_nan || reml_bad {
                println!("  >>> {label}: SILENTLY WRONG FIT (non-finite output, no error)");
            }
        }
        Ok(_) => println!("{label}: OK non-standard fit"),
        Err(e) => println!("{label}: ERR (clean) {e:?}"),
    }
}

#[test]
fn diag_numrobust() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(7);
    let unit = Uniform::new(0.0_f64, 1.0).expect("u");
    let noise = Normal::new(0.0, 0.3).expect("n");

    // Baseline sane data.
    let n = 200usize;
    let base_x: Vec<f64> = (0..n).map(|_| unit.sample(&mut rng)).collect();
    let base_y: Vec<f64> = base_x
        .iter()
        .map(|x| (6.0 * x).sin() + noise.sample(&mut rng))
        .collect();

    // AXIS 1: extreme y scale (1e8) and tiny y scale (1e-8).
    let y_big: Vec<f64> = base_y.iter().map(|y| y * 1e8).collect();
    summarize("y_scale_1e8", &fit_from_formula("y ~ s(x)", &dataset_xy(base_x.clone(), y_big), &cfg));
    let y_small: Vec<f64> = base_y.iter().map(|y| y * 1e-8).collect();
    summarize("y_scale_1e-8", &fit_from_formula("y ~ s(x)", &dataset_xy(base_x.clone(), y_small), &cfg));

    // AXIS 2: extreme x scale.
    let x_big: Vec<f64> = base_x.iter().map(|x| x * 1e8).collect();
    summarize("x_scale_1e8", &fit_from_formula("y ~ s(x)", &dataset_xy(x_big, base_y.clone()), &cfg));
    let x_small: Vec<f64> = base_x.iter().map(|x| x * 1e-8).collect();
    summarize("x_scale_1e-8", &fit_from_formula("y ~ s(x)", &dataset_xy(x_small, base_y.clone()), &cfg));

    // AXIS 3: zero-variance y (all identical).
    let y_const: Vec<f64> = vec![3.5; n];
    summarize("y_zero_variance", &fit_from_formula("y ~ s(x)", &dataset_xy(base_x.clone(), y_const), &cfg));

    // AXIS 4: single unique x (collinear / degenerate design).
    let x_const: Vec<f64> = vec![0.5; n];
    summarize("x_single_value", &fit_from_formula("y ~ s(x)", &dataset_xy(x_const, base_y.clone()), &cfg));

    // AXIS 5: tiny n.
    summarize("n2", &fit_from_formula("y ~ s(x)", &dataset_xy(vec![0.1, 0.9], vec![1.0, 2.0]), &cfg));
    summarize("n3", &fit_from_formula("y ~ s(x)", &dataset_xy(vec![0.1, 0.5, 0.9], vec![1.0, 0.5, 2.0]), &cfg));

    // AXIS 6: x with a single huge outlier (heavy leverage).
    let mut x_out = base_x.clone();
    x_out[0] = 1e12;
    summarize("x_outlier_1e12", &fit_from_formula("y ~ s(x)", &dataset_xy(x_out, base_y.clone()), &cfg));

    // AXIS 7: y with one Inf/NaN — should be a CLEAN error not a panic/silent.
    let mut y_inf = base_y.clone();
    y_inf[0] = f64::INFINITY;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_from_formula("y ~ s(x)", &dataset_xy(base_x.clone(), y_inf), &cfg)
    }));
    match r {
        Ok(res) => summarize("y_has_inf", &res),
        Err(_) => println!("y_has_inf: >>> PANIC (should be clean error)"),
    }

    panic!("diag_numrobust intentional fail to surface stdout");
}
