//! End-to-end quality: gam's **Poisson(log) × tensor-product** smooth must
//! match `mgcv` — the mature, standard tensor-product GAM reference — on the
//! exact same data.
//!
//! This benchmarks the *critical cross-feature combination* that family-
//! agnostic basis code most often gets wrong: a non-Gaussian family (Poisson
//! with the canonical log link) layered on top of a multi-dimensional
//! tensor-product `te()` smooth. A 1-D Gaussian smooth can pass while the
//! tensor + IRLS-row interaction is subtly broken, so we test them together:
//!   * gam fits `y ~ te(x, z, k=[6,6])`, family = poisson, REML.
//!   * mgcv fits `gam(y ~ te(x, z, bs="ps", k=c(6,6)), family=poisson(link=
//!     "log"), method="REML")` — its tensor-product machinery is the field
//!     reference.
//!
//! Basis alignment matters for a *tight* comparison: gam's `te()` uses cubic
//! (degree-3) B-spline margins with a 2nd-order difference penalty per margin
//! (term_builder: `degree=3`, `penalty_order=2`, single per-margin penalty).
//! mgcv's default `te()` margin is thin-plate (`bs="tp"`), which would inject a
//! pure basis-convention gap. We therefore pin mgcv to `bs="ps"` (P-splines):
//! its default `m=c(2,2)` gives exactly a cubic B-spline basis with a 2nd-order
//! penalty, the *same* marginal construction as gam. With `k=6` per margin both
//! engines build 6 basis functions per axis (gam: internal_knots =
//! max(6-(3+1),1) = 2, total = 2+3+1 = 6), so the marginal bases coincide.
//!
//! Both engines then target the *same* penalized REML objective on the *same*
//! data, so their fitted Poisson means (response scale, exp(eta)) must track
//! pointwise across the whole surface. We assert relative L2 on the fitted
//! means. A genuine divergence here is a real bug in gam's PIRLS-row /
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
use rand_distr::{Distribution, Poisson};

/// Deterministic 15×20 Poisson surface (n=300): x on an even grid over [0,2π],
/// z on an even grid over [-1,1], log-mean eta = 0.8 + 0.3*sin(x) + 0.2*z^2,
/// y ~ Poisson(exp(eta)) with the Poisson draws seeded (seed=345). The grid is
/// fully deterministic and only the response carries noise, so the identical
/// (x, z, y) triples reach both gam and mgcv via the shared CSV the harness
/// writes — there is no sampling difference between the two engines.
fn make_poisson_tensor_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nx = 15usize;
    let nz = 20usize;
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut y = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        // even grid endpoints included: x in [0, 2π], z in [-1, 1].
        let xi = (ix as f64) / ((nx - 1) as f64) * (2.0 * std::f64::consts::PI);
        for iz in 0..nz {
            let zi = -1.0 + 2.0 * (iz as f64) / ((nz - 1) as f64);
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

    // ---- fit with gam: y ~ te(x, z, k=[6,6]), poisson(log), REML ----------
    // k=6 per margin: cubic B-spline (degree 3) requires k >= 4; 6 leaves room
    // to express the sin(x) / z^2 structure and matches the mgcv ps margins.
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam poisson te fit");
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
    // bs="ps" with default m=c(2,2) => cubic B-spline margins + 2nd-order
    // penalty, matching gam's te() margin construction (see module doc).
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, bs = "ps", k = c(6, 6)), data = df,
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
        "poisson te(x,z,bs=ps,k=c(6,6)): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5}"
    );

    // Both engines REML-fit the identical Poisson surface with *matched* cubic
    // B-spline (P-spline) 6×6 tensor margins and a 2nd-order marginal penalty,
    // so the only remaining gap is each engine's independent REML lambda search
    // and tensor reparameterization on one noisy Poisson draw. That keeps the
    // fitted means very close: rel_l2 < 0.06 is tight enough to catch any real
    // break in the PIRLS-row / tensor-design integration this combination
    // exercises, while leaving headroom for the legitimate lambda-search
    // difference. pearson > 0.99 guards the surface shape independently.
    assert!(
        corr > 0.99,
        "Poisson+te() fitted means decorrelate from mgcv: pearson={corr:.5} (rel_l2={rel:.4})"
    );
    assert!(
        rel < 0.06,
        "Poisson+te() fitted means diverge from mgcv: rel_l2={rel:.4} (pearson={corr:.5})"
    );
    // EDF is reparameterization/null-space-convention sensitive across the two
    // tensor implementations, so assert same-ballpark complexity rather than
    // bit-identical: within 35% relative.
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);
    assert!(
        edf_rel < 0.35,
        "tensor effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
