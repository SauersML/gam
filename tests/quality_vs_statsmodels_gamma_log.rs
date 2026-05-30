//! End-to-end OBJECTIVE quality: gam's Gamma(log) GLM must RECOVER the known
//! truth function on positive continuous data — not merely reproduce another
//! tool's fit.
//!
//! Capability under test: `family = "gamma"` (ResponseFamily::Gamma with a log
//! inverse link, see src/solver/workflow.rs) fit through the additive smooth
//! machinery `y ~ s(x, k=5) + s(z, k=4)`. The Gamma family is the standard GLM
//! for strictly-positive continuous outcomes with multiplicative error (cost /
//! severity / survival-time data); the log link is universal for it.
//!
//! OBJECTIVE METRIC (truth recovery): the data are simulated from a KNOWN log-
//! mean surface eta_true = 2.0 + 0.5*sin(x*pi/5) + 0.3*cos(z*pi/6), with
//! y ~ Gamma(shape=2, scale=exp(eta_true)/2) so E[y] = exp(eta_true). Because
//! the Gamma error is multiplicative (Var = mu^2/shape), the natural scale-free
//! accuracy measure is RMSE on the LOG-MEAN predictor: how well does gam's fitted
//! eta_hat = log(mu_hat) track the true eta? The PRIMARY assertion is that gam
//! recovers that surface — RMSE(eta_hat, eta_true) is a small fraction of the
//! signal's amplitude (and far below the per-observation noise sd of the working
//! log-response). This is a real quality claim about gam, independent of any peer
//! tool.
//!
//! BASELINE TO MATCH-OR-BEAT: statsmodels `GLM(y, X, family=Gamma(link=Log()))`
//! is fit on gam's identical frozen basis (so the only difference is gam's mild
//! k=5/k=4 wiggliness penalty). We additionally require gam's truth-recovery
//! error to be no worse than statsmodels' by more than 10% — i.e. on the
//! objective accuracy metric, gam matches or beats the mature reference. We still
//! print rel_l2(mu) / pearson(eta) vs statsmodels for context, but neither is a
//! pass criterion.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
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
    let mut eta_true = Vec::<f64>::with_capacity(n);
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
        eta_true.push(eta);
    }
    assert!(
        y.iter().all(|&v| v > 0.0),
        "Gamma outcomes must be positive"
    );

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
    let result = fit_from_formula("y ~ s(x, k=5) + s(z, k=4)", &ds, &cfg).expect("gam gamma fit");
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

    // ---- OBJECTIVE quality: truth recovery on the log-mean surface --------
    // gam's eta_hat = log(mu_hat) is on the same scale as the simulated eta_true;
    // the Gamma's multiplicative error makes log-scale RMSE the natural scale-free
    // accuracy measure. statsmodels, fit on the SAME frozen basis, gives the
    // unpenalized-MLE error on this surface as a match-or-beat baseline.
    let gam_rmse_truth = rmse(&gam_eta, &eta_true);
    let ref_rmse_truth = rmse(ref_eta, &eta_true);

    // Signal amplitude of the wiggly part (range of 0.5*sin(.) + 0.3*cos(.) is
    // [-0.8, 0.8] => 1.6) and the per-observation noise on the working log-
    // response (Var(log y) for Gamma(shape) is the trigamma psi'(shape); for
    // shape = 2, psi'(2) = pi^2/6 - 1 ~= 0.6449, so noise sd ~= 0.803). A fitted
    // log-mean within ~0.25 RMSE of truth is well inside the signal range and
    // roughly a third of the per-point noise sd — a genuine recovery, not a fit
    // tracking noise.
    let signal_range = 1.6_f64;
    let working_noise_sd = (std::f64::consts::PI * std::f64::consts::PI / 6.0 - 1.0).sqrt();
    let truth_bar = 0.25_f64;

    // Context only (NOT a pass criterion): agreement with statsmodels' own fit.
    let rel = relative_l2(&gam_mu, ref_mu);
    let corr = pearson(&gam_eta, ref_eta);
    let sm_scale = r.scalar("scale");

    eprintln!(
        "gamma(log) s(x,k=5)+s(z,k=4): n={n} p={p} gam_edf={gam_edf:.3} \
         sm_scale={sm_scale:.4} signal_range={signal_range:.3} \
         working_noise_sd={working_noise_sd:.3} \
         rmse(eta_hat,eta_true)=gam:{gam_rmse_truth:.5} sm:{ref_rmse_truth:.5} \
         [context only] rel_l2(mu)={rel:.5} pearson(eta)={corr:.5}"
    );

    // PRIMARY claim: gam recovers the true log-mean surface.
    assert!(
        gam_rmse_truth < truth_bar,
        "Gamma(log) failed to recover the true log-mean surface: \
         rmse(eta_hat, eta_true)={gam_rmse_truth:.5} >= bar {truth_bar:.5} \
         (signal_range={signal_range:.3}, working_noise_sd={working_noise_sd:.3})"
    );

    // MATCH-OR-BEAT on accuracy: gam's truth-recovery error must not exceed the
    // mature reference's by more than 10%. (Gamma's beta-MLE is independent of
    // scale/shape, so on the shared frozen basis statsmodels is the unpenalized
    // optimum; gam's mild penalty trades a little fidelity for less variance and
    // must stay within this margin.)
    assert!(
        gam_rmse_truth <= ref_rmse_truth * 1.10,
        "Gamma(log) truth-recovery error worse than statsmodels by >10%: \
         gam={gam_rmse_truth:.5} vs statsmodels={ref_rmse_truth:.5}"
    );
}
