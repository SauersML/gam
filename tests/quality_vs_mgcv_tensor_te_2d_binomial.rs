//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **binomial** family (logit link) must match `mgcv` — the mature, standard
//! GAM implementation — on identical aggregated binomial-count data.
//!
//! Binomial is the second-most-common applied family after Gaussian, and a
//! tensor smooth crossed with a non-Gaussian family is the acid test for gam's
//! GLM infrastructure outside the Poisson case: the logit link inversion
//! (`mu = 1/(1+e^{-eta})`) and the binomial working weight (`prior_weight *
//! mu*(1-mu)`) both run inside the PIRLS reweight loop, so a mishandled link
//! gradient or variance term shows up as a divergence from mgcv here.
//!
//! Both engines fit the SAME model by REML on the SAME data:
//!   * mgcv : `gam(cbind(success_count, failure_count) ~ te(x, z, k = 6),
//!            family = binomial(link = "logit"), method = "REML")`
//!   * gam  : `prop ~ te(x, z, k = 6)`, family `binomial`, link `logit`, with
//!            a per-row trial-count `weight_column` — the standard GLM encoding
//!            of `cbind(successes, failures)` as (proportion, prior weight =
//!            trials).
//!
//! We compare the fitted **linear predictor on the logit scale** (the quantity
//! the smoother actually estimates, before link inversion) element-wise at the
//! 250 training points. gam's `design * beta` is exactly that logit-scale eta;
//! mgcv's `predict(type = "link")` is the same thing. Both minimize the same
//! penalized binomial deviance under REML, so the etas must essentially
//! coincide; a real divergence is a real bug in gam's binomial PIRLS.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
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

    // ---- synthetic aggregated-binomial data (seed=20260530) ---------------
    // x, z ~ U[0,1]^2; true logit-scale surface is a separable tensor signal
    // pi/4 * (sin(2*pi*x) + cos(2*pi*z)). For each row, n_trials=20 Bernoulli
    // trials are summed into success_count (the exact integer counts handed to
    // BOTH engines, so there is zero data-encoding skew between them).
    const N: usize = 250;
    const N_TRIALS: u32 = 20;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut success_count = Vec::with_capacity(N);
    let mut failure_count = Vec::with_capacity(N);
    let mut prop = Vec::with_capacity(N);
    let mut trials = Vec::with_capacity(N);

    for _ in 0..N {
        let xi = u01.sample(&mut rng);
        let zi = u01.sample(&mut rng);
        let surface = (PI / 4.0) * ((TAU * xi).sin() + (TAU * zi).cos());
        let p = invlogit(surface);
        // Sum N_TRIALS independent Bernoulli(p) draws into an integer count.
        let mut s: u32 = 0;
        for _ in 0..N_TRIALS {
            if u01.sample(&mut rng) < p {
                s += 1;
            }
        }
        let succ = s as f64;
        let fail = (N_TRIALS - s) as f64;
        x.push(xi);
        z.push(zi);
        success_count.push(succ);
        failure_count.push(fail);
        prop.push(succ / N_TRIALS as f64);
        trials.push(N_TRIALS as f64);
    }

    // ---- fit with gam: prop ~ te(x, z, k=6), binomial/logit, REML ---------
    // gam encodes the binomial counts as proportion response + trial weights,
    // the standard GLM `cbind(successes, failures)` representation.
    let headers: Vec<String> = ["x", "z", "prop", "trials"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        rows.push(csv::StringRecord::from(vec![
            x[i].to_string(),
            z[i].to_string(),
            prop[i].to_string(),
            trials[i].to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        weight_column: Some("trials".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("prop ~ te(x, z, k=6)", &ds, &cfg).expect("gam binomial te fit");
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

    // ---- compare on the logit (linear-predictor) scale --------------------
    let rel = relative_l2(&gam_eta, mgcv_eta);
    let corr = pearson(&gam_eta, mgcv_eta);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "te(x,z) binomial/logit: n={N} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2(eta)={rel:.4} pearson(eta)={corr:.5} edf_rel={edf_rel:.3}"
    );

    // Both engines REML-fit the same penalized binomial deviance on identical
    // integer counts, so their logit-scale linear predictors must essentially
    // coincide. The bound is slightly looser than the Gaussian-lidar smooth
    // (rel<0.02): the binomial link inversion + mu*(1-mu) working weight add
    // genuine numerical sensitivity at extreme eta, where p saturates toward 0/1
    // and the working weight collapses, so per-row eta estimates can wander a
    // little more than in the identity-link Gaussian case. rel_l2 < 0.035 and
    // pearson > 0.985 still demand near-coincident surfaces and would catch any
    // real binomial-PIRLS divergence (wrong working weight, mis-signed link
    // gradient, or a botched logit inversion all blow these up well past it).
    assert!(
        corr > 0.985,
        "binomial te(x,z) linear predictors should be near-identical: pearson={corr:.5}"
    );
    assert!(
        rel < 0.035,
        "binomial te(x,z) linear predictors diverge from mgcv: rel_l2={rel:.4}"
    );
    // Same-ballpark model complexity (basis/null-space conventions differ between
    // gam's tensor margins and mgcv's k=6 te), within 30% relative.
    assert!(
        edf_rel < 0.30,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
