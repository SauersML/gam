//! End-to-end **objective quality**: does gam's 2-D Poisson smooth recover the
//! *true* log-mean surface it was generated from?
//!
//! The data are simulated from a known generator
//!   `true_eta(x, z) = sin(pi*x) * cos(pi*z)`,  `count ~ Poisson(exp(true_eta))`,
//! so we have ground truth on the linear-predictor (log) scale. The quality
//! claim is **truth recovery**, asserted directly against `true_eta`:
//!
//!   * PRIMARY (accuracy): `RMSE(gam_eta, true_eta)` is small in absolute terms.
//!     The signal spans `[-1, 1]` (range 2); we require the recovered surface to
//!     sit well inside a small fraction of that range for both the additive
//!     `s(x) + s(z)` model and the tensor `te(x, z)` model. This proves gam's
//!     Poisson family / log-link PIRLS and its tensor-penalty construction
//!     actually estimate the surface, not merely that they imitate a peer tool.
//!
//!   * BASELINE (match-or-beat): pyGAM — an independent, mature GAM engine with
//!     its own scipy/scikit-learn bases and PIRLS fit — is fit on the *identical*
//!     counts. We require gam's recovery error to be no worse than pyGAM's by
//!     more than 10% (`gam_rmse <= pygam_rmse * 1.10`). pyGAM is here only as a
//!     yardstick on the SAME objective metric (distance to truth); we never
//!     assert that gam reproduces pyGAM's (itself noisy) fit.
//!
//! We deliberately do NOT assert that gam's effective degrees of freedom match
//! pyGAM's — matching another tool's complexity proves nothing. We only sanity-
//! check that gam's edf lands in a signal-appropriate range (above a straight
//! line, below the basis dimension), which is an objective structural property.
//!
//! The reference rel_l2 / pearson against pyGAM are still computed and printed
//! for context via `eprintln!`, but they are NOT pass/fail criteria.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

#[test]
fn gam_poisson_2d_recovers_true_log_mean_surface() {
    init_parallelism();

    // ---- synthetic Poisson-count truth on the unit square ------------------
    // true_eta = sin(pi*x) * cos(pi*z); count ~ Poisson(exp(true_eta)).
    // Same seed and same draw order as quality_vs_mgcv_tensor_te_2d_poisson.rs,
    // so the SAME counts feed gam and pyGAM; both are scored against true_eta.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut count = Vec::with_capacity(n);
    let mut true_eta = Vec::with_capacity(n);
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
        true_eta.push(eta);
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

    // ---- fit BOTH models with pyGAM (independent baseline on the SAME data) -
    // PoissonGAM(s(0,n_splines=6)+s(1,n_splines=6)) is the additive analog;
    // PoissonGAM(te(0,1,n_splines=6)) is the tensor-product analog. predict_mu
    // returns the mean (mu = exp(eta) under the log link), so eta = log(mu) is
    // the linear predictor on the same (log) scale gam reports and on which the
    // truth `true_eta` lives. pyGAM is scored against the SAME truth as gam.
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

    // ---- OBJECTIVE metric: recovery error against the KNOWN truth ----------
    // Score every fit by RMSE to `true_eta`; pyGAM is scored the same way so it
    // is a baseline-to-beat on the SAME objective metric, not a fit to imitate.
    let gam_add_rmse = rmse(&gam_add_eta, &true_eta);
    let pygam_add_rmse = rmse(pygam_add_eta, &true_eta);
    let gam_te_rmse = rmse(&gam_te_eta, &true_eta);
    let pygam_te_rmse = rmse(pygam_te_eta, &true_eta);

    // Context-only (NOT pass/fail): how close gam's fit is to pyGAM's fit.
    let add_rel = relative_l2(&gam_add_eta, pygam_add_eta);
    let add_corr = pearson(&gam_add_eta, pygam_add_eta);
    let te_rel = relative_l2(&gam_te_eta, pygam_te_eta);
    let te_corr = pearson(&gam_te_eta, pygam_te_eta);

    eprintln!(
        "Poisson 2-D additive: n={n} gam_edf={gam_add_edf:.3} pygam_edf={pygam_add_edf:.3} \
         gam_rmse(truth)={gam_add_rmse:.4} pygam_rmse(truth)={pygam_add_rmse:.4} \
         [context: rel_l2(gam,pygam)={add_rel:.4} pearson={add_corr:.5}]"
    );
    eprintln!(
        "Poisson 2-D tensor:   n={n} gam_edf={gam_te_edf:.3} pygam_edf={pygam_te_edf:.3} \
         gam_rmse(truth)={gam_te_rmse:.4} pygam_rmse(truth)={pygam_te_rmse:.4} \
         [context: rel_l2(gam,pygam)={te_rel:.4} pearson={te_corr:.5}]"
    );

    // ---- PRIMARY assertion: absolute truth recovery ------------------------
    // The signal sin(pi*x)*cos(pi*z) spans [-1, 1] (range 2). With n=300 Poisson
    // counts whose mean is exp(eta) in [exp(-1), exp(1)] ~= [0.37, 2.72], the
    // information per point is modest, so a faithful smoother recovers the
    // log-mean surface to a small fraction of the signal range. We require the
    // pointwise RMSE on the log scale to stay under 0.30 (15% of the 2.0 range)
    // for the additive model and under 0.40 (20%) for the tensor model, which is
    // far below the noise a botched link / penalty would inject and well within
    // what a correct Poisson GAM achieves at this sample size.
    assert!(
        gam_add_rmse < 0.30,
        "additive Poisson fit does not recover the true log-mean surface: \
         rmse(gam, truth)={gam_add_rmse:.4} (>= 0.30)"
    );
    assert!(
        gam_te_rmse < 0.40,
        "tensor Poisson fit does not recover the true log-mean surface: \
         rmse(gam, truth)={gam_te_rmse:.4} (>= 0.40)"
    );

    // ---- BASELINE: match-or-beat pyGAM on the SAME accuracy metric ---------
    // gam's recovery error must be no worse than pyGAM's by more than 10%.
    assert!(
        gam_add_rmse <= pygam_add_rmse * 1.10,
        "additive Poisson: gam's recovery error exceeds pyGAM's by >10%: \
         gam_rmse={gam_add_rmse:.4} pygam_rmse={pygam_add_rmse:.4}"
    );
    assert!(
        gam_te_rmse <= pygam_te_rmse * 1.10,
        "tensor Poisson: gam's recovery error exceeds pyGAM's by >10%: \
         gam_rmse={gam_te_rmse:.4} pygam_rmse={pygam_te_rmse:.4}"
    );

    // ---- STRUCTURE: edf in a sane, signal-appropriate range ----------------
    // Not matched to pyGAM. Each model has two k=6 marginal bases, so the basis
    // dimension is ~11 (additive: 2*(6-1)+1) / ~36 (tensor: ~6*6). A real 2-D
    // signal must use more than a flat line (edf > 1) and far less than the full
    // basis (well below the basis dimension), which falsifies both a collapsed
    // (over-smoothed to constant) and a saturated (interpolating) fit.
    assert!(
        gam_add_edf > 1.0 && gam_add_edf < 11.0,
        "additive edf outside the signal-appropriate range (1, 11): {gam_add_edf:.3}"
    );
    assert!(
        gam_te_edf > 1.0 && gam_te_edf < 36.0,
        "tensor edf outside the signal-appropriate range (1, 36): {gam_te_edf:.3}"
    );
}
