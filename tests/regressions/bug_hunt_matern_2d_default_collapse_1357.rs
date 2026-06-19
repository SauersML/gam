// Regression for #1357: the default isotropic 2-D Matern radial smooth
// `y ~ matern(x1, x2)` collapsed to a degenerate constant fit (EDF ~1, flat
// surface, R^2 ~ 0) on the large majority of ordinary data draws, even though
// `te(x1, x2)` recovers the same signal perfectly. The collapse came from the
// isotropic-kappa outer optimizer parking at a degenerate large-length-scale
// corner where the kernel block is flat, so REML then shrank the whole smooth
// away.
//
// DGP (from the issue): x1, x2 ~ U(-1, 1), n = 400, truth =
// sin(2*x1) + cos(1.5*x2) + 0.5*x1*x2, y = truth + N(0, 0.1). The fit must
// recover a non-degenerate surface (EDF well above the intercept-only floor
// and R^2 against the noise-free truth comfortably positive) on the seeds the
// issue flagged as failing.

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

fn truth(x1: f64, x2: f64) -> f64 {
    (2.0 * x1).sin() + (1.5 * x2).cos() + 0.5 * x1 * x2
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(-1.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x1 = u.sample(&mut rng);
            let x2 = u.sample(&mut rng);
            let y = truth(x1, x2) + noise.sample(&mut rng);
            StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// R^2 of `matern(x1, x2)` against the noise-free truth on a fixed eval grid,
/// plus the fitted total EDF.
fn matern_fit_quality(seed: u64) -> (f64, f64) {
    let data = build_dataset(400, 0.1, seed);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ matern(x1, x2)", &data, &cfg)
        .unwrap_or_else(|e| panic!("matern(x1, x2) failed to fit (seed {seed}): {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for matern(x1, x2)");
    };
    let edf = fit.fit.edf_total().expect("edf_total present");

    // Interior eval grid in [-0.9, 0.9]^2 (avoid the boundary where any radial
    // smooth extrapolates).
    let g: Vec<f64> = (0..20).map(|i| -0.9 + 1.8 * i as f64 / 19.0).collect();
    let m = g.len();
    let mut design_in = Array2::<f64>::zeros((m * m, 3));
    let mut truth_vals = Vec::with_capacity(m * m);
    let mut row = 0;
    for &a in &g {
        for &b in &g {
            design_in[[row, 0]] = a;
            design_in[[row, 1]] = b;
            design_in[[row, 2]] = 0.0;
            truth_vals.push(truth(a, b));
            row += 1;
        }
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild matern design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "matern predictions must all be finite (seed {seed})"
    );

    let mean_t = truth_vals.iter().sum::<f64>() / truth_vals.len() as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(truth_vals.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = truth_vals.iter().map(|t| (t - mean_t).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    (r2, edf)
}

#[test]
fn matern_2d_default_recovers_signal_on_ordinary_draws() {
    init_parallelism();
    // Seeds the issue reports as collapsing (the median draw). Before the fix
    // these landed at EDF ~1 with R^2 ~ 0; after the fix they recover the
    // smooth surface.
    for seed in [2_u64, 3, 4, 6, 7] {
        let (r2, edf) = matern_fit_quality(seed);
        assert!(
            edf > 2.0,
            "matern(x1, x2) collapsed to the intercept-only floor (seed {seed}): edf={edf:.3}"
        );
        assert!(
            r2 > 0.8,
            "matern(x1, x2) failed to recover the signal (seed {seed}): R^2={r2:.4}, edf={edf:.3}"
        );
    }
}
