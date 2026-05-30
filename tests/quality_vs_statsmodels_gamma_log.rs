//! End-to-end quality: gam's Gamma(log) GLM must match statsmodels — the mature,
//! standard general-GLM implementation — on positive continuous data.
//!
//! Capability under test: `family = "gamma"` (ResponseFamily::Gamma with a log
//! inverse link, see src/solver/workflow.rs) fit through the additive smooth
//! machinery `y ~ s(x, k=5) + s(z, k=4)`. The Gamma family is the standard GLM
//! for strictly-positive continuous outcomes with multiplicative error (cost /
//! severity / survival-time data); the log link is universal for it.
//!
//! Reference: statsmodels `GLM(y, X, family=Gamma(link=Log()))`. To make this a
//! clean apples-to-apples GLM check — isolating gam's Gamma likelihood / scale /
//! working-response machinery rather than confounding it with two engines'
//! independent smoothing-parameter selection — gam first fits and selects its
//! smoothing parameters, then we evaluate gam's *frozen* penalized basis at the
//! data points and hand that exact design matrix to statsmodels' Gamma GLM.
//! Both engines then maximise the identical Gamma log-likelihood over the
//! identical column space; gam additionally applies a *mild* wiggliness penalty,
//! but the smooths are deliberately low-rank (k = 5 and k = 4), so they are
//! nearly saturated and the penalty barely bites. Their fitted means must
//! therefore essentially coincide. A real divergence here is a real bug in gam's
//! Gamma working-response / scale path, not a basis mismatch.
//!
//! We compare:
//!   1. relative L2 of the fitted means mu = exp(eta) (the quantity Gamma cares
//!      about — its variance is mu^2/shape), and
//!   2. Pearson correlation of the log-scale fitted predictors eta = log(mu).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};

#[test]
fn gam_gamma_log_matches_statsmodels() {
    init_parallelism();

    // ---- synthetic positive-continuous data (canonical Gamma parametrization) ----
    // truth: eta = 2.0 + 0.5*sin(x*pi/5) + 0.3*cos(z*pi/6)
    // y ~ Gamma(shape=2, scale=exp(eta)/2)  =>  E[y] = shape*scale = exp(eta).
    // seed = 456, n = 180, x,z ~ U(0,10).
    let n = 180usize;
    let seed = 456u64;
    let shape = 2.0_f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 10.0).expect("uniform 0..10");

    let mut x = Vec::<f64>::with_capacity(n);
    let mut z = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let zi = ux.sample(&mut rng);
        let eta = 2.0
            + 0.5 * (xi * std::f64::consts::PI / 5.0).sin()
            + 0.3 * (zi * std::f64::consts::PI / 6.0).cos();
        let scale = eta.exp() / shape; // shape*scale = exp(eta) = E[y]
        let g = Gamma::new(shape, scale).expect("gamma(shape,scale)");
        let yi = g.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    assert!(y.iter().all(|&v| v > 0.0), "Gamma outcomes must be positive");

    // ---- encode for gam ---------------------------------------------------
    let headers: Vec<String> = ["y", "x", "z"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string(), z[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: Gamma(log), y ~ s(x,k=5) + s(z,k=4) ----------------
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=5) + s(z, k=4)", &ds, &cfg).expect("gam gamma fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma(log)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild gam's frozen design at the training points; with a log link the
    // linear predictor is eta = design*beta and the fitted mean is mu = exp(eta).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let p = design.design.ncols();
    assert_eq!(design.design.nrows(), n, "design row count must match data");

    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // Materialize gam's design densely so we can hand the *exact* same basis to
    // statsmodels: column j is design * e_j. The basis already carries gam's
    // intercept/constant column, so statsmodels fits the GLM with no extra
    // constant. We transmit the design as p separate equal-length columns
    // d0..d{p-1} (the harness requires equal-length columns — no flattened
    // ragged layout) plus a constant "p" column so the Python body knows the
    // column count. statsmodels maximises the identical Gamma log-likelihood
    // (mu = exp(X beta)) over the identical column space.
    let design_cols: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let mut e = Array1::<f64>::zeros(p);
            e[j] = 1.0;
            design.design.apply(&e).to_vec()
        })
        .collect();
    let p_col = vec![p as f64; n];

    let mut columns: Vec<Column<'_>> = Vec::with_capacity(p + 2);
    columns.push(Column::new("y", &y));
    columns.push(Column::new("p", &p_col));
    let col_names: Vec<String> = (0..p).map(|j| format!("d{j}")).collect();
    for j in 0..p {
        columns.push(Column::new(&col_names[j], &design_cols[j]));
    }

    let r = run_python(
        &columns,
        r#"
import numpy as np
import statsmodels.api as sm

n = len(df["y"])
p = int(df["p"][0])
X = np.column_stack([np.asarray(df["d%d" % j], dtype=float) for j in range(p)])
yv = np.asarray(df["y"], dtype=float)

# gam's basis already includes its own intercept column, so do NOT add one.
model = sm.GLM(yv, X, family=sm.families.Gamma(link=sm.families.links.Log()))
res = model.fit()
eta = X @ res.params
mu = np.exp(eta)
emit("mu", mu)
emit("eta", eta)
emit("scale", [res.scale])
"#,
    );
    let ref_mu = r.vector("mu");
    let ref_eta = r.vector("eta");
    assert_eq!(ref_mu.len(), n, "statsmodels mu length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_mu, ref_mu);
    let corr = pearson(&gam_eta, ref_eta);
    let sm_scale = r.scalar("scale");

    eprintln!(
        "gamma(log) s(x,k=5)+s(z,k=4): n={n} p={p} gam_edf={gam_edf:.3} \
         sm_scale={sm_scale:.4} rel_l2(mu)={rel:.5} pearson(eta)={corr:.5}"
    );

    // Both engines maximise the identical Gamma(log) log-likelihood over the
    // identical (frozen) basis; Gamma's MLE for beta is independent of the
    // shared scale/shape, so the fits would coincide exactly but for gam's mild
    // k=5/k=4 wiggliness penalty (worth a few percent on the wiggly part of eta).
    // rel_l2 < 0.10 on mu and pearson > 0.99 on log-scale eta is the principled
    // bound: it allows for that light penalty wedge yet is tight enough that any
    // genuine Gamma working-response/dispersion/link bug fails it.
    assert!(
        corr > 0.99,
        "Gamma(log) log-scale predictors diverge from statsmodels: pearson(eta)={corr:.5}"
    );
    assert!(
        rel < 0.10,
        "Gamma(log) fitted means diverge from statsmodels: rel_l2(mu)={rel:.5}"
    );
}
