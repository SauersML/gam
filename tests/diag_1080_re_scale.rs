//! TEMP diagnostic for #1080: fit the synthetic RE+by-smooth model (no R) and
//! print gam's per-block edf, lambdas, sigma, plus a directly recomputed RSS
//! from the rebuilt design, to localize the residual-variance bias.
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

const N_GROUPS: usize = 5;
const PER_GROUP: usize = 60;
const SEED: u64 = 88;
const RESID_SD: f64 = 0.2;

fn base_smooth(x: f64) -> f64 {
    (3.0 * std::f64::consts::PI * x).sin()
}

#[test]
fn diag_1080_re_scale() {
    init_parallelism();
    let n = N_GROUPS * PER_GROUP;
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let slope_dist = Normal::new(0.0, 0.5).expect("slope normal");
    let intercept_dist = Normal::new(0.0, 0.4).expect("intercept normal");
    let noise = Normal::new(0.0, RESID_SD).expect("noise normal");
    let slope_g: Vec<f64> = (0..N_GROUPS).map(|_| slope_dist.sample(&mut rng)).collect();
    let intercept_g: Vec<f64> = (0..N_GROUPS)
        .map(|_| intercept_dist.sample(&mut rng))
        .collect();

    let mut x = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    let mut true_mean = Vec::<f64>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n);
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let mui = base_smooth(xi) + 0.15 * slope_g[grp] * xi + intercept_g[grp];
            let yi = mui + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            true_mean.push(mui);
            g_code.push(grp as f64);
            rows.push(StringRecord::from(vec![
                format!("{xi}"),
                format!("g{grp}"),
                format!("{yi}"),
            ]));
        }
    }
    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x) + s(x, by=g) + group(g)", &ds, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("standard fit");
    };
    let edf_total = fit.fit.edf_total().expect("edf");
    let sigma = fit.fit.standard_deviation;

    // rebuild design at training points and recompute RSS directly
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, g_idx]] = g_code[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let rss: f64 = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| (yi - fi) * (yi - fi))
        .sum();
    let rmse_truth: f64 =
        (true_mean
            .iter()
            .zip(fitted.iter())
            .map(|(&t, &f)| (t - f) * (t - f))
            .sum::<f64>()
            / n as f64)
            .sqrt();

    eprintln!("DIAG#1080:");
    eprintln!("  edf_total={edf_total:.4}");
    eprintln!("  sigma={sigma:.6}  sigma^2(internal)={:.6}", sigma * sigma);
    eprintln!("  rebuilt RSS={rss:.4}  RSS/(n-edf)={:.6}", rss / (n as f64 - edf_total));
    eprintln!("  rebuilt RSS/n={:.6}", rss / n as f64);
    eprintln!("  rmse(fitted,truth)={rmse_truth:.6}");
    eprintln!("  edf_by_block={:?}", fit.fit.edf_by_block());
    eprintln!("  lambdas={:?}", fit.fit.lambdas);
    eprintln!("  beta.len()={}", fit.fit.beta.len());
}
