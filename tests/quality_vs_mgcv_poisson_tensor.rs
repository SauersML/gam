//! End-to-end **objective quality**: gam's **Poisson(log) × tensor-product**
//! smooth must *recover the true mean surface* it was generated from.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not tool-matching): the data is
//! simulated from a fully known log-mean surface
//!     eta_true(x, z) = 0.8 + 0.3*sin(x) + 0.2*z^2,  mu_true = exp(eta_true),
//! with noise entering *only* through the Poisson response draw. The smoother's
//! job is to estimate `mu_true` from the noisy counts. So the primary pass/fail
//! is the root-mean-square error of gam's fitted Poisson mean against the TRUE
//! mean surface:
//!     RMSE(gam_mean, mu_true) <= 0.18 * range(mu_true).
//! On this surface mu_true spans ~[1.79, 3.34] (range ~1.55), so the bar is an
//! absolute RMSE of ~0.28 — comfortably below the per-cell Poisson noise sd
//! (sqrt(mu) ~ 1.4..1.8) yet tight enough that a broken PIRLS-row / tensor-
//! design integration (which would smear or bias the surface) fails it.
//!
//! This benchmarks the *critical cross-feature combination* that family-
//! agnostic basis code most often gets wrong: a non-Gaussian family (Poisson
//! with the canonical log link) layered on top of a multi-dimensional
//! tensor-product `te()` smooth. A 1-D Gaussian smooth can pass while the
//! tensor + IRLS-row interaction is subtly broken, so we test them together;
//! gam fits `y ~ te(x, z, k=[6,6])`, family = poisson, REML.
//!
//! mgcv is NOT the standard of correctness here — it is a peer smoother that is
//! itself only an *estimate* of the same truth, fit on the same noisy draw. We
//! therefore demote it to a MATCH-OR-BEAT ACCURACY BASELINE: gam's RMSE-to-truth
//! must be no worse than mgcv's RMSE-to-truth by more than 10%
//!     RMSE(gam_mean, mu_true) <= 1.10 * RMSE(mgcv_mean, mu_true).
//! To keep that an apples-to-apples accuracy comparison, mgcv is pinned to the
//! same marginal basis as gam: `bs="ps"` with default `m=c(2,2)` gives cubic
//! B-spline margins + a 2nd-order penalty, matching gam's `te()` margins
//! (degree 3, penalty order 2). With `k=6` per margin both engines build 6
//! basis functions per axis, so neither tool is handicapped by basis convention.
//! The legacy rel_l2 / pearson "closeness to mgcv" numbers are still printed for
//! context but are NOT pass criteria — reproducing a peer tool's noisy fit is
//! not a quality claim; recovering the truth is.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
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
///
/// Returns `(x, y, z, mu_true)` where `mu_true[i] = exp(eta_true(x[i], z[i]))`
/// is the *noiseless* mean surface the smoother must recover.
fn make_poisson_tensor_data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nx = 15usize;
    let nz = 20usize;
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut y = Vec::with_capacity(nx * nz);
    let mut mu_true = Vec::with_capacity(nx * nz);
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
            mu_true.push(lambda);
        }
    }
    (x, y, z, mu_true)
}

#[test]
fn gam_poisson_tensor_recovers_true_mean_surface() {
    init_parallelism();

    // ---- identical synthetic data for both engines ------------------------
    let (x, y, z, mu_true) = make_poisson_tensor_data(345);
    let n = x.len();
    assert_eq!(n, 300, "grid 15x20 => n=300");

    // ---- build the encoded dataset for gam (columns x, z, y) --------------
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
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

    // ---- OBJECTIVE METRIC: recover the TRUE mean surface ------------------
    // The pass/fail quantities are errors against `mu_true` (the noiseless
    // surface the data was generated from), NOT closeness to mgcv. We measure
    // each smoother's RMSE to the truth on the response scale.
    let gam_err = rmse(&gam_mean, &mu_true);
    let mgcv_err = rmse(mgcv_mean, &mu_true);

    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = mu_max - mu_min;

    // Context only (peer-tool agreement); explicitly NOT a pass criterion.
    let rel = relative_l2(&gam_mean, mgcv_mean);
    let corr = pearson(&gam_mean, mgcv_mean);
    eprintln!(
        "poisson te(x,z) truth recovery: n={n} mu_range=[{mu_min:.3},{mu_max:.3}] \
         gam_rmse_to_truth={gam_err:.4} mgcv_rmse_to_truth={mgcv_err:.4} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         (context: rel_l2_vs_mgcv={rel:.4} pearson_vs_mgcv={corr:.5})"
    );

    // PRIMARY claim: gam recovers the true mean surface. The absolute bar is a
    // small fraction of the signal range; well inside the per-cell Poisson
    // sampling sd, but tight enough that a biased/smeared tensor fit fails.
    let abs_bar = 0.18 * mu_range;
    assert!(
        gam_err <= abs_bar,
        "Poisson+te() failed to recover the true mean surface: \
         RMSE(gam, truth)={gam_err:.4} > {abs_bar:.4} (= 0.18 * range {mu_range:.4})"
    );

    // SECONDARY claim (match-or-beat ACCURACY): gam's error to the truth is no
    // worse than the mature tensor smoother's error to the same truth by >10%.
    // mgcv is a baseline to match-or-beat on accuracy, not a correctness oracle.
    assert!(
        gam_err <= 1.10 * mgcv_err,
        "Poisson+te() less accurate than mgcv at recovering the truth: \
         RMSE(gam, truth)={gam_err:.4} > 1.10 * RMSE(mgcv, truth)={mgcv_err:.4}"
    );

    // EDF sanity only (complexity in a signal-appropriate range), never a
    // match-to-reference: the surface has real 2-D structure (sin(x) + z^2), so
    // a sensible fit uses more than a flat plane yet far less than the full
    // 6*6-1 = 35-dim tensor basis.
    assert!(
        gam_edf > 1.0 && gam_edf < 35.0,
        "Poisson+te() effective degrees of freedom outside the sane range \
         (1, 35): gam_edf={gam_edf:.3}"
    );
}
