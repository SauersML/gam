//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under a
//! **non-Gaussian** family (Poisson, log link) must match `mgcv` — the mature,
//! standard GAM implementation — on identical data.
//!
//! This is the essential *combination* test: tensor products under a
//! non-Gaussian family. A Gaussian tensor smooth can fit perfectly while a
//! Poisson one diverges if the tensor-product penalty is recomputed incorrectly
//! across PIRLS iterations or the log-link gradient/weights are mishandled. Both
//! engines fit `count ~ te(x, z, k=7)` with `family = poisson(link = "log")` by
//! REML, so they target the same penalized log-likelihood and the recovered
//! *linear predictor* (log-mean surface) must coincide.
//!
//! We compare on the **linear-predictor (log) scale** — the scale on which the
//! tensor penalty actually acts and the natural place to detect a PIRLS /
//! link-inversion bug — evaluated at the shared training points, asserting:
//!   1. relative L2 of the log-mean surfaces is tiny, and
//!   2. the two surfaces are essentially perfectly correlated.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

use gam::test_support::reference::{Column, pearson, relative_l2, run_r};

#[test]
fn gam_tensor_te_2d_poisson_matches_mgcv() {
    init_parallelism();

    // ---- synthetic Poisson-count truth on the unit square ------------------
    // count_expected = exp(sin(pi*x) * cos(pi*z)); count ~ Poisson(count_expected).
    // Fixed seed => the SAME draws feed gam and mgcv, so any disagreement is in
    // the fitting, not the data.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut count = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = (PI * xi).sin() * (PI * zi).cos(); // true log-mean
        let lambda = eta.exp().max(1e-12);
        let draw: f64 = Poisson::new(lambda).expect("valid Poisson rate").sample(&mut rng);
        x.push(xi);
        z.push(zi);
        count.push(draw);
    }

    // ---- fit with gam: count ~ te(x, z, k=7), Poisson / log link, REML ------
    let headers = ["x", "z", "count"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                z[i].to_string(),
                count[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("count ~ te(x, z, k=7)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(x, z)");
    };

    // gam linear predictor (log scale) at the training points: rebuilding the
    // frozen design and applying beta yields eta = design*beta directly, BEFORE
    // the log-link inverse — exactly the scale on which the tensor penalty acts.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // family = poisson(link = "log"), method = "REML"; emit the linear predictor.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("count", &count),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(count ~ te(x, z, k = 7), data = df,
                 family = poisson(link = "log"), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), n, "mgcv linear-predictor length mismatch");

    // ---- compare on the log (linear-predictor) scale -----------------------
    let rel = relative_l2(&gam_eta, mgcv_eta);
    let corr = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) Poisson/log: n={n} mgcv_edf={mgcv_edf:.3} rel_l2(eta)={rel:.4} pearson(eta)={corr:.5}"
    );

    // Both engines REML-fit the identical Poisson data through the same log link
    // with a rank-7-per-margin tensor product, so the recovered log-mean surfaces
    // must essentially coincide. The spec bound (rel_l2 < 0.03, pearson > 0.99)
    // is tight enough that a per-iteration penalty mis-application or a botched
    // link-gradient — which would distort the surface well beyond a few percent —
    // is caught, while leaving margin for benign basis/centering-convention
    // differences between gam and mgcv.
    assert!(
        corr > 0.99,
        "Poisson tensor log-mean surfaces should be near-identical: pearson={corr:.5}"
    );
    assert!(
        rel < 0.03,
        "Poisson tensor log-mean surface diverges from mgcv: rel_l2={rel:.4}"
    );
}
