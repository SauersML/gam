//! End-to-end quality: gam's **Poisson(log) × tensor-product** smooth must
//! match `mgcv` — the mature, standard tensor-product GAM reference — on the
//! exact same data.
//!
//! This benchmarks the *critical cross-feature combination* that family-
//! agnostic basis code most often gets wrong: a non-Gaussian family (Poisson
//! with the canonical log link) layered on top of a multi-dimensional
//! tensor-product `te()` smooth. A 1-D Gaussian smooth can pass while the
//! tensor + IRLS-row interaction is subtly broken, so we test them together:
//!   * gam fits `y ~ te(x, z, k=[4,3])`, family = poisson, REML.
//!   * mgcv fits `gam(y ~ te(x, z, k = c(4, 3)), family = poisson(link="log"),
//!     method = "REML")` — its tensor-product machinery is the field reference.
//! Both engines target the *same* penalized REML objective on the *same* data,
//! so their fitted Poisson means (on the response scale, exp(eta)) must track
//! pointwise across the entire 15×20 surface. We assert relative L2 on those
//! fitted means. A genuine divergence here is a real bug in gam's PIRLS-row /
//! tensor-design integration, not a reason to loosen the bound.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};

/// Synthetic Poisson surface on a 15×20 grid (n=300): x~U(0,2π), z~U(-1,1),
/// log-mean eta = 0.8 + 0.3*sin(x) + 0.2*z^2, y ~ Poisson(exp(eta)). Seed=345,
/// fed *identically* to gam and mgcv via the shared CSV the harness writes.
fn make_poisson_tensor_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 2.0 * std::f64::consts::PI).expect("uniform x");
    let uz = Uniform::new(-1.0, 1.0).expect("uniform z");
    let nx = 15usize;
    let nz = 20usize;
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut y = Vec::with_capacity(nx * nz);
    for _ in 0..nx {
        for _ in 0..nz {
            let xi = ux.sample(&mut rng);
            let zi = uz.sample(&mut rng);
            let eta = 0.8 + 0.3 * xi.sin() + 0.2 * zi * zi;
            let lambda = eta.exp();
            let pois = Poisson::new(lambda).expect("poisson lambda > 0");
            let yi: f64 = pois.sample(&mut rng);
            x.push(xi);
            z.push(zi);
            y.push(yi);
        }
    }
    (x, y, z)
}

#[test]
fn gam_poisson_tensor_matches_mgcv() {
    init_parallelism();

    // ---- identical synthetic data for both engines ------------------------
    let (x, y, z) = make_poisson_tensor_data(345);
    let n = x.len();
    assert_eq!(n, 300, "grid 15x20 => n=300");

    // ---- build the encoded dataset for gam (columns x, z, y) --------------
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                x[i].to_string(),
                z[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: y ~ te(x, z, k=[4,3]), poisson(log), REML ----------
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[4,3])", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for poisson(log) + te()");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted means on the response scale: rebuild the tensor design at the
    // observed (x, z), then mean = exp(design * beta) under the log link.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild tensor design at training points");
    let gam_eta = design.design.apply(&fit.fit.beta);
    let gam_mean: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv (the mature tensor reference) -------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, k = c(4, 3)), data = df,
                 family = poisson(link = "log"), method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_mean = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_mean.len(), n, "mgcv fitted length mismatch");

    // ---- compare fitted Poisson means on the response (exp(eta)) scale ----
    let rel = relative_l2(&gam_mean, mgcv_mean);
    let corr = pearson(&gam_mean, mgcv_mean);
    eprintln!(
        "poisson te(x,z,k=c(4,3)): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5}"
    );

    // Both engines REML-fit the identical Poisson surface with a 4×3 tensor
    // basis, so the fitted means must coincide closely. The residual gap is the
    // B-spline (gam margins) vs thin-plate (mgcv te default) marginal-basis
    // convention plus REML lambda-selection differences on a single noisy draw;
    // rel_l2 < 0.12 is tight enough that any real break in the PIRLS-row /
    // tensor-design integration (the thing this combination exercises) fails it,
    // while not penalizing the legitimate basis-convention difference.
    assert!(
        rel < 0.12,
        "Poisson+te() fitted means diverge from mgcv: rel_l2={rel:.4} (pearson={corr:.5})"
    );
}
