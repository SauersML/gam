//! End-to-end quality: gam's 1-D **binomial(logit) smooth** shape recovery,
//! benchmarked on the **logit (linear-predictor) scale** against **pyGAM's
//! `LogisticGAM`** — the second-most-directly-comparable Python GAM engine
//! (after statsmodels) for penalized binomial GAMs.
//!
//! Both gam and pyGAM fit a penalized binomial GAM by PIRLS over a 1-D smooth
//! basis: gam selects its smoothing parameter by REML, pyGAM by a grid search
//! minimizing (generalized) cross-validation / UBRE. Isolating a single 1-D
//! smooth `s(x)` strips away everything but the smooth shape-recovery logic, so
//! if the fitted *logit-scale* eta diverges it points straight at gam's binomial
//! reweight (the IRLS working weights `mu*(1-mu)`) or the penalty application —
//! a real bug, not a basis-convention footnote.
//!
//! We compare on the **linear-predictor (logit) scale** deliberately: the probit
//! / probability scale compresses extreme eta through the squashing inverse-link
//! and would mask divergence in the tails. eta is where the smoother actually
//! lives.
//!
//! Data is a fixed-seed synthetic 1-D problem (n=200, x in [0,100]) whose truth
//! is a smooth sinusoid on the logit scale, `eta_truth = 0.3*sin(pi*x/50)`, with
//! Bernoulli responses. The *identical* x and y vectors are fed to both engines.
//!
//! Bounds (principled, un-weakened): `pearson(eta_gam, eta_pygam) > 0.98` and
//! `rel_l2(eta) < 0.06` — the L2 budget absorbs the REML-vs-GCV smoothing-
//! parameter slack on a 200-row binary problem (binary data carries little
//! information per point, so the two lambda criteria pick visibly different
//! amounts of smoothing) while still falsifying any genuine reweight/penalty
//! bug, and EDF agreement within 25% asserts same-ballpark model complexity
//! across the two penalty/lambda conventions.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;

#[test]
fn gam_logistic_1d_shape_matches_pygam_on_logit_scale() {
    init_parallelism();

    // ---- fixed-seed synthetic 1-D logistic problem ------------------------
    // x uniform on [0,100]; truth on the logit scale is a smooth sinusoid that
    // completes one full period over the domain; y ~ Bernoulli(logistic(eta)).
    // A self-contained, fully reproducible LCG (no external RNG crate) makes the
    // x and y vectors byte-identical for both engines.
    let n = 200usize;
    let mut state: u64 = 42; // seed=42
    let mut next_unit = || -> f64 {
        // SplitMix64-style advance; map to [0,1).
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let truth_eta = |xi: f64| 0.3 * (std::f64::consts::PI * xi / 50.0).sin();
    for i in 0..n {
        let xi = 100.0 * next_unit();
        let p = 1.0 / (1.0 + (-truth_eta(xi)).exp());
        let yi = if next_unit() < p { 1.0 } else { 0.0 };
        x[i] = xi;
        y[i] = yi;
    }
    // Sanity: both classes present (required for a meaningful logistic fit).
    let n_pos = y.iter().filter(|&&v| v > 0.5).count();
    assert!(
        n_pos > 10 && n_pos < n - 10,
        "synthetic data should have both classes well represented, got {n_pos}/{n} positives"
    );

    // ---- materialize the synthetic data as CSV and load it via gam --------
    // (load_csvwith_inferred_schema is the same loader the canonical reference
    // tests use; a temp file keeps the data path identical to those tests.)
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{}\n", x[i], y[i] as i64));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_pygam_logistic_1d_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic logistic data");
    std::fs::remove_file(&tmp).ok();
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: y ~ s(x, k=10), binomial / logit / REML ------------
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam binomial(logit) fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) 1-D smooth should be a Standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate the fitted smooth on a dense, sorted grid spanning the domain so
    // the shape comparison is independent of the random sampling density; the
    // logit-scale eta is exactly design*beta (the link is applied afterwards).
    let n_grid = 100usize;
    let mut xg = vec![0.0f64; n_grid];
    let mut grid = Array2::<f64>::zeros((n_grid, ds.headers.len()));
    for j in 0..n_grid {
        let xj = 100.0 * (j as f64) / (n_grid as f64 - 1.0);
        xg[j] = xj;
        grid[[j, x_idx]] = xj;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 1-D smooth design on evaluation grid");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_eta.len(), n_grid, "gam eta grid length mismatch");

    // ---- fit the SAME model with pyGAM's LogisticGAM (the reference) -------
    // LogisticGAM(s(0, n_splines=10)) is a penalized binomial PIRLS fit over a
    // cubic B-spline smooth; we predict the linear predictor on the identical
    // grid. pyGAM exposes the logit-scale predictor either via predict_mu's
    // inverse or the link directly; we compute it from the predicted prob to
    // avoid relying on a private attribute, then return eta = logit(mu).
    let py = run_python(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            // The evaluation grid travels as a (padded) column so both engines
            // see the exact same query points; only the first n_grid entries are
            // meaningful (n_grid < n), the tail is ignored on the Python side.
            Column::new("xg", &{
                let mut g = vec![0.0f64; n];
                g[..n_grid].copy_from_slice(&xg);
                g
            }),
        ],
        r#"
from pygam import LogisticGAM, s
X = np.asarray(df["x"], dtype=float).reshape(-1, 1)
yv = np.asarray(df["y"], dtype=float)
n_grid = 100
Xg = np.asarray(df["xg"], dtype=float)[:n_grid].reshape(-1, 1)
gam = LogisticGAM(s(0, n_splines=10)).fit(X, yv)
mu = np.asarray(gam.predict_mu(Xg), dtype=float)
# logit-scale linear predictor; clip mu off the open-interval boundary so the
# logit is finite even when a grid point sits in a near-saturated region.
mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
eta = np.log(mu / (1.0 - mu))
emit("eta", eta)
emit("edf", [float(gam.statistics_["edof"])])
"#,
    );
    let pygam_eta = py.vector("eta");
    let pygam_edf = py.scalar("edf");
    assert_eq!(pygam_eta.len(), n_grid, "pyGAM eta grid length mismatch");

    // ---- compare on the logit scale ---------------------------------------
    let corr = pearson(&gam_eta, pygam_eta);
    let rel = relative_l2(&gam_eta, pygam_eta);
    let edf_rel = (gam_edf - pygam_edf).abs() / pygam_edf.abs().max(1.0);

    eprintln!(
        "synthetic logistic s(x,k=10): n={n} n_pos={n_pos} gam_edf={gam_edf:.3} \
         pygam_edf={pygam_edf:.3} (edf_rel={edf_rel:.3}) \
         eta pearson={corr:.5} rel_l2={rel:.4}"
    );

    // (1) Shape agreement on the logit scale is the load-bearing claim: two
    // independent penalized binomial PIRLS engines must trace essentially the
    // same linear predictor. A wrong binomial working weight or misapplied
    // penalty decorrelates eta well below this threshold.
    assert!(
        corr > 0.98,
        "gam vs pyGAM logit-scale eta shapes diverge: pearson={corr:.5}"
    );
    // (2) L2 budget absorbs the REML-vs-GCV lambda-selection slack on a small
    // binary sample; 0.06 is loose enough not to flag that slack yet tight
    // enough to catch a real reweight/penalty bug.
    assert!(
        rel < 0.06,
        "gam binomial smooth diverges from pyGAM on the logit scale: rel_l2={rel:.4}"
    );
    // (3) EDF same-ballpark model complexity across differing penalty/lambda
    // conventions: within 25% relative (the spec bound).
    assert!(
        edf_rel < 0.25,
        "effective degrees of freedom disagree: gam={gam_edf:.3} pygam={pygam_edf:.3} (rel={edf_rel:.3})"
    );
}
