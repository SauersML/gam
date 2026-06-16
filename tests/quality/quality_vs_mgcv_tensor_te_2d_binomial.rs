//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **binomial** family (logit link) must RECOVER the true logit-scale surface
//! the data was generated from.
//!
//! OBJECTIVE METRIC (the pass/fail claim): the data is synthesised from a known
//! separable surface `f(x,z) = (pi/4) * (sin(2*pi*x) + cos(2*pi*z))` on the
//! logit scale, then corrupted by binomial sampling (20 Bernoulli trials per
//! row). We assert that gam's fitted logit-scale linear predictor recovers that
//! TRUTH: `RMSE(gam_eta, f_true)` must be small relative to the surface's own
//! amplitude (the per-row signal range), i.e. the smooth explains the bulk of
//! the true structure rather than overfitting the binomial noise. This is an
//! objective accuracy claim about gam, not a "same output as mgcv" claim.
//!
//! Binomial is the second-most-common applied family after Gaussian, and a
//! tensor smooth crossed with a non-Gaussian family is the acid test for gam's
//! GLM infrastructure outside the Poisson case: the logit link inversion
//! (`mu = 1/(1+e^{-eta})`) and the binomial working weight (`prior_weight *
//! mu*(1-mu)`) both run inside the PIRLS reweight loop. A mishandled link
//! gradient or variance term inflates the recovery error well past the bar.
//!
//! mgcv is retained only as a BASELINE TO MATCH-OR-BEAT on that same recovery
//! metric: we fit the identical model in mgcv and require gam's truth-recovery
//! RMSE to be no worse than mgcv's by more than 10%. Matching mgcv's noisy fit
//! is NOT the claim — both engines are scored against the analytic truth, and
//! gam must recover it at least as accurately as the mature reference.
//!
//! Both engines fit the SAME underlying data by REML. gam's `binomial` family
//! is Bernoulli (it requires y in {0,1}), so the binomial trials are handed to
//! gam EXPANDED: one row per individual Bernoulli draw carrying the binary
//! outcome y in {0,1} at that row's (x, z). mgcv fits the mathematically
//! identical likelihood from the AGGREGATED integer counts at each design point
//! (`cbind(successes, failures)` at the N unique (x, z) points). For a binomial
//! GLM the aggregated-count fit and the expanded-Bernoulli fit are the same MLE
//! / penalized fit, so there is no data-encoding skew between the engines:
//!   * mgcv : `gam(cbind(success_count, failure_count) ~ te(x, z, k = 6),
//!            family = binomial(link = "logit"), method = "REML")` on N rows
//!   * gam  : `y ~ te(x, z, k = 6)`, family `binomial`, link `logit`, on the
//!            N * N_TRIALS expanded binary rows.
//! Both engines are then evaluated on the SAME N unique design points and scored
//! against the SAME analytic logit-scale truth at those points.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

fn invlogit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

#[test]
fn gam_tensor_te_2d_binomial_logit_matches_mgcv() {
    init_parallelism();

    // ---- synthetic binomial data (seed=20260530) --------------------------
    // x, z ~ U[0,1]^2; true logit-scale surface is a separable tensor signal
    // pi/4 * (sin(2*pi*x) + cos(2*pi*z)). For each of N design points, n_trials
    // = 20 independent Bernoulli(p) draws are taken. The AGGREGATED success
    // counts go to mgcv via cbind(); the SAME individual binary outcomes go to
    // gam as one expanded row per trial (gam's binomial family is Bernoulli and
    // requires y in {0,1}). The two encodings yield the identical binomial
    // likelihood, so there is no data skew between the engines.
    const N: usize = 250;
    const N_TRIALS: u32 = 20;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    // Per-design-point arrays (length N): used for mgcv's aggregated fit and for
    // the truth-recovery metric, which is scored at the N unique design points.
    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut success_count = Vec::with_capacity(N);
    let mut failure_count = Vec::with_capacity(N);
    // The known logit-scale truth at each design point — the quantity gam must
    // recover. Stored exactly as generated so the recovery metric is honest.
    let mut f_true = Vec::with_capacity(N);

    // Expanded per-trial arrays (length N * N_TRIALS): the binary data gam
    // consumes — one Bernoulli outcome y in {0,1} per row at that point's (x,z).
    let n_expanded = N * N_TRIALS as usize;
    let mut x_exp = Vec::with_capacity(n_expanded);
    let mut z_exp = Vec::with_capacity(n_expanded);
    let mut y_exp = Vec::with_capacity(n_expanded);

    for _ in 0..N {
        let xi = u01.sample(&mut rng);
        let zi = u01.sample(&mut rng);
        let surface = (PI / 4.0) * ((TAU * xi).sin() + (TAU * zi).cos());
        let p = invlogit(surface);
        // Draw N_TRIALS independent Bernoulli(p) outcomes; record each binary
        // outcome (for gam) and accumulate the integer success count (for mgcv).
        let mut s: u32 = 0;
        for _ in 0..N_TRIALS {
            let yi = if u01.sample(&mut rng) < p { 1.0 } else { 0.0 };
            s += yi as u32;
            x_exp.push(xi);
            z_exp.push(zi);
            y_exp.push(yi);
        }
        let succ = s as f64;
        let fail = (N_TRIALS - s) as f64;
        x.push(xi);
        z.push(zi);
        success_count.push(succ);
        failure_count.push(fail);
        f_true.push(surface);
    }

    // ---- fit with gam: y ~ te(x, z, k=6), binomial/logit, REML ------------
    // gam's binomial family is Bernoulli, so it receives the EXPANDED binary
    // outcomes (one row per Bernoulli trial), not aggregated proportions.
    let headers: Vec<String> = ["x", "z", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_expanded);
    for i in 0..n_expanded {
        rows.push(csv::StringRecord::from(vec![
            x_exp[i].to_string(),
            z_exp[i].to_string(),
            y_exp[i].to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=6)", &ds, &cfg).expect("gam binomial te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial/logit te(x,z)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted linear predictor (logit scale) at the training points:
    // rebuild the frozen design at (x,z) and apply beta. With a logit link the
    // design*beta IS the linear predictor eta (the link is applied separately).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te(x,z) design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_eta.len(), N, "gam eta length mismatch");

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("success_count", &success_count),
            Column::new("failure_count", &failure_count),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(cbind(success_count, failure_count) ~ te(x, z, k = 6),
                 family = binomial(link = "logit"), data = df, method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), N, "mgcv eta length mismatch");

    // ---- OBJECTIVE METRIC: recovery of the known logit-scale surface ------
    // The smooth (in both gam and mgcv) carries a free intercept, so an overall
    // additive level is unidentifiable from the shape we care about. Score every
    // surface in the SAME mean-centered frame against the centered truth, so the
    // metric measures recovered structure, not an arbitrary constant offset.
    fn mean_center(v: &[f64]) -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect()
    }
    let truth_c = mean_center(&f_true);
    let gam_c = mean_center(&gam_eta);
    let mgcv_c = mean_center(mgcv_eta);

    // Truth-recovery error of each engine against the analytic surface.
    let gam_rmse = rmse(&gam_c, &truth_c);
    let mgcv_rmse = rmse(&mgcv_c, &truth_c);

    // Signal scale: peak-to-peak amplitude of the centered true surface. The
    // generating surface has logit-scale range up to pi/2 ~= 1.57 each margin,
    // so the centered truth spans well over 1.5 in eta; an RMSE that is a small
    // fraction of that means the structure is genuinely recovered, not the noise.
    let truth_range = truth_c.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - truth_c.iter().cloned().fold(f64::INFINITY, f64::min);

    // For context only (NOT a pass criterion): how close gam lands to mgcv.
    let rel = relative_l2(&gam_c, &mgcv_c);
    let corr = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) binomial/logit recovery: n={N} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         truth_range={truth_range:.3} gam_rmse_vs_truth={gam_rmse:.4} \
         mgcv_rmse_vs_truth={mgcv_rmse:.4} | context rel_l2(gam,mgcv)={rel:.4} pearson={corr:.5}"
    );

    // PRIMARY CLAIM: gam recovers the true surface. The binomial sampling noise
    // on 20-trial counts injects per-row logit noise of order
    // 1/sqrt(20*p*(1-p)) ~ 0.45-0.5 at mid-p; a properly-penalized smooth must
    // average that down to a recovery RMSE far below the ~3.0 peak-to-peak signal
    // amplitude. We demand the error be under 12% of the signal range — a smooth
    // that overfit the noise or mis-handled the logit working weight would sit
    // well above this.
    let recovery_bar = 0.12 * truth_range;
    assert!(
        gam_rmse <= recovery_bar,
        "gam failed to recover the true logit surface: rmse_vs_truth={gam_rmse:.4} \
         > {recovery_bar:.4} (= 0.12 * signal range {truth_range:.3})"
    );

    // BASELINE TO MATCH-OR-BEAT: gam's recovery accuracy must be no worse than
    // the mature reference's by more than 10%. This scores BOTH engines against
    // the analytic truth — not gam against mgcv's fit — so passing means gam is
    // at least as good at recovering the signal as mgcv, never merely that it
    // imitates mgcv's noisy output.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam recovers the true surface worse than mgcv: gam_rmse={gam_rmse:.4} \
         > 1.10 * mgcv_rmse={mgcv_rmse:.4}"
    );

    // Sanity on complexity: gam's effective dof must sit in a signal-appropriate
    // range — strictly above a flat fit (1) and below the basis dimension. A
    // k=6 te(x,z) tensor has up to 36 coefficients; an edf that collapsed to ~1
    // (oversmoothed away the structure) or saturated near the basis size
    // (interpolated the noise) would both signal a broken fit.
    assert!(
        gam_edf > 1.5 && gam_edf < 30.0,
        "gam effective dof out of signal-appropriate range: edf={gam_edf:.3}"
    );
}
