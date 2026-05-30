//! End-to-end quality: gam's 2-D Poisson smooth shape (linear predictor on the
//! log scale) cross-checked against **pyGAM** — an independent, mature GAM
//! library built on scipy / scikit-learn bases with its own PIRLS fit. pyGAM's
//! `PoissonGAM` supports both additive `s()` smooths and `te()` tensor-product
//! ("product") smooths under the Poisson family with a log link, which makes it
//! a rare *second* GAM engine that exposes both shapes head-to-head with gam.
//!
//! Why this is the right comparator and why two paths:
//!   * **Additive path** (`count ~ s(x) + s(z)` both sides): a direct head-to-
//!     head on the same model class. Both engines build penalized B-spline
//!     margins, invert the same log link in PIRLS, and target the same penalized
//!     Poisson log-likelihood. The recovered log-mean surface must coincide
//!     tightly — this isolates the Poisson family / link-gradient logic from any
//!     tensor-penalty subtlety.
//!   * **Tensor path** (`count ~ te(x, z)` both sides): adds the multiplicative
//!     basis structure of a product smooth. gam's tensor penalty (a Kronecker
//!     sum of marginal penalties with per-margin smoothing parameters selected
//!     by REML) and pyGAM's tensor `te()` (a single GCV-selected penalty over
//!     the product basis) are *not* identical penalty constructions, so the
//!     bound is looser here — but the recovered log-mean surface must still
//!     agree in shape, which falsifies any per-iteration tensor-penalty
//!     mis-application or botched log-link inversion.
//!
//! We compare on the **linear-predictor (log) scale** — `eta = log(mu)` for both
//! engines — at the shared training points. That is the scale on which the
//! penalty acts and the natural place to catch a PIRLS / link-inversion bug.
//!
//! Why looser bounds than the mgcv tests: pyGAM selects its smoothing
//! parameter(s) by a grid search minimizing GCV under a PIRLS fit, whereas gam
//! selects lambda by REML. The two criteria pick slightly different amounts of
//! smoothing on the same data, so the curves agree in shape (correlation) very
//! tightly but can differ by a few percent in L2. The additive bounds
//! (pearson > 0.995, rel_l2 < 0.05) are a near head-to-head; the tensor bounds
//! (pearson > 0.98, rel_l2 < 0.08) absorb the genuinely different tensor-penalty
//! parametrizations while still falsifying a real divergence in the smoother or
//! family logic. EDF agreement is asserted within 25% (different penalty
//! normalizations + lambda-selection criteria).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

#[test]
fn gam_poisson_2d_shape_matches_pygam() {
    init_parallelism();

    // ---- synthetic Poisson-count truth on the unit square ------------------
    // true_eta = sin(pi*x) * cos(pi*z); count ~ Poisson(exp(true_eta)).
    // Same seed and same draw order as quality_vs_mgcv_tensor_te_2d_poisson.rs,
    // so the SAME counts feed gam and pyGAM and any disagreement is in the fit.
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
        let draw: f64 = Poisson::new(lambda)
            .expect("valid Poisson rate")
            .sample(&mut rng);
        x.push(xi);
        z.push(zi);
        count.push(draw);
    }

    // ---- encode the shared dataset for gam ---------------------------------
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

    // Shared grid (training points) used to rebuild gam's frozen design and
    // recover its linear predictor eta = design*beta (BEFORE the log-link
    // inverse) for both the additive and tensor fits.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    // ---- gam additive: count ~ s(x, k=6) + s(z, k=6), Poisson/log, REML ----
    let add_result = fit_from_formula("count ~ s(x, k=6) + s(z, k=6)", &ds, &cfg)
        .expect("gam poisson additive fit");
    let FitResult::Standard(add_fit) = add_result else {
        panic!("expected a standard GAM fit for the additive Poisson model");
    };
    let gam_add_edf = add_fit.fit.edf_total().expect("gam reports additive edf");
    let add_design = build_term_collection_design(grid.view(), &add_fit.resolvedspec)
        .expect("rebuild additive design at training points");
    let gam_add_eta: Vec<f64> = add_design.design.apply(&add_fit.fit.beta).to_vec();

    // ---- gam tensor: count ~ te(x, z, k=6), Poisson/log, REML --------------
    let te_result =
        fit_from_formula("count ~ te(x, z, k=6)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(te_fit) = te_result else {
        panic!("expected a standard GAM fit for the tensor Poisson model");
    };
    let gam_te_edf = te_fit.fit.edf_total().expect("gam reports tensor edf");
    let te_design = build_term_collection_design(grid.view(), &te_fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_te_eta: Vec<f64> = te_design.design.apply(&te_fit.fit.beta).to_vec();

    // ---- fit BOTH models with pyGAM (the independent GAM reference) --------
    // PoissonGAM(s(0,n_splines=6)+s(1,n_splines=6)) is the additive head-to-head;
    // PoissonGAM(te(0,1,n_splines=6)) is the tensor-product analog. predict_mu
    // returns the mean (mu = exp(eta) under the log link), so eta = log(mu) is
    // the linear predictor on the same scale gam reports.
    let py = run_python(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("count", &count),
        ],
        r#"
from pygam import PoissonGAM, s, te
X = np.column_stack([
    np.asarray(df["x"], dtype=float),
    np.asarray(df["z"], dtype=float),
])
y = np.asarray(df["count"], dtype=float)

add = PoissonGAM(s(0, n_splines=6) + s(1, n_splines=6)).fit(X, y)
add_mu = np.asarray(add.predict_mu(X), dtype=float)
emit("add_eta", np.log(np.clip(add_mu, 1e-12, None)))
emit("add_edf", [float(add.statistics_["edof"])])

ten = PoissonGAM(te(0, 1, n_splines=6)).fit(X, y)
ten_mu = np.asarray(ten.predict_mu(X), dtype=float)
emit("te_eta", np.log(np.clip(ten_mu, 1e-12, None)))
emit("te_edf", [float(ten.statistics_["edof"])])
"#,
    );
    let pygam_add_eta = py.vector("add_eta");
    let pygam_add_edf = py.scalar("add_edf");
    let pygam_te_eta = py.vector("te_eta");
    let pygam_te_edf = py.scalar("te_edf");
    assert_eq!(pygam_add_eta.len(), n, "pyGAM additive eta length mismatch");
    assert_eq!(pygam_te_eta.len(), n, "pyGAM tensor eta length mismatch");

    // ---- compare on the log (linear-predictor) scale -----------------------
    let add_rel = relative_l2(&gam_add_eta, pygam_add_eta);
    let add_corr = pearson(&gam_add_eta, pygam_add_eta);
    let add_edf_rel = (gam_add_edf - pygam_add_edf).abs() / pygam_add_edf.abs().max(1.0);

    let te_rel = relative_l2(&gam_te_eta, pygam_te_eta);
    let te_corr = pearson(&gam_te_eta, pygam_te_eta);
    let te_edf_rel = (gam_te_edf - pygam_te_edf).abs() / pygam_te_edf.abs().max(1.0);

    eprintln!(
        "pyGAM Poisson 2-D additive: n={n} gam_edf={gam_add_edf:.3} pygam_edf={pygam_add_edf:.3} \
         rel_l2(eta)={add_rel:.4} pearson(eta)={add_corr:.5} edf_rel={add_edf_rel:.3}"
    );
    eprintln!(
        "pyGAM Poisson 2-D tensor:   n={n} gam_edf={gam_te_edf:.3} pygam_edf={pygam_te_edf:.3} \
         rel_l2(eta)={te_rel:.4} pearson(eta)={te_corr:.5} edf_rel={te_edf_rel:.3}"
    );

    // ---- additive path: near head-to-head (same model class + log link) ----
    // Two independent GAM engines on the same additive B-spline basis, both
    // PIRLS-inverting the log link on identical Poisson data, must trace the
    // same log-mean surface; only the REML-vs-GCV lambda choice separates them.
    assert!(
        add_corr > 0.995,
        "additive Poisson log-mean surfaces diverge from pyGAM: pearson={add_corr:.5}"
    );
    assert!(
        add_rel < 0.05,
        "additive Poisson log-mean surface diverges from pyGAM: rel_l2={add_rel:.4}"
    );

    // ---- tensor path: looser bound for genuinely different penalties -------
    // gam's tensor penalty (Kronecker sum, per-margin REML lambdas) and pyGAM's
    // single GCV-selected product-basis penalty are not the same construction,
    // so we allow more L2 slack and a slightly lower correlation floor; 0.98 /
    // 0.08 still catch a per-iteration tensor-penalty mis-application or a
    // botched log-link inversion (either distorts the surface far beyond this).
    assert!(
        te_corr > 0.98,
        "tensor Poisson log-mean surfaces diverge from pyGAM: pearson={te_corr:.5}"
    );
    assert!(
        te_rel < 0.08,
        "tensor Poisson log-mean surface diverges from pyGAM: rel_l2={te_rel:.4}"
    );

    // ---- EDF agreement: same-ballpark model complexity within 25% ----------
    // Penalty normalization + lambda-selection criteria differ, so we assert
    // comparable effective degrees of freedom rather than bit-identical.
    assert!(
        add_edf_rel < 0.25,
        "additive effective degrees of freedom disagree: gam={gam_add_edf:.3} \
         pygam={pygam_add_edf:.3} (rel={add_edf_rel:.3})"
    );
    assert!(
        te_edf_rel < 0.25,
        "tensor effective degrees of freedom disagree: gam={gam_te_edf:.3} \
         pygam={pygam_te_edf:.3} (rel={te_edf_rel:.3})"
    );
}
