//! End-to-end OBJECTIVE quality: gam's **latent-cloglog binomial** family — the
//! exact marginal complementary-log-log model under a latent Gaussian frailty
//! `u ~ N(0, σ²)` — must RECOVER the true marginal probability surface of
//! frailty-overdispersed binary data, and do so better than a plain (frailty-
//! ignoring) cloglog GAM.
//!
//! Why this test earns its place. The latent-cloglog family
//! (`family = "latent-cloglog-binomial"`) had no end-to-end FIT quality test —
//! only an inverse-link jet-correctness unit test. Its marginal mean is
//!   μ(η) = E_u[ 1 − exp(−exp(η + σ·u)) ],   u ~ N(0,1),
//! the proper model when a binary outcome carries an unobserved Gaussian frailty:
//! the frailty attenuates the population-level (marginal) link relative to the
//! conditional cloglog, so a plain cloglog GAM, which has no frailty term, fits a
//! biased mean. gam's latent-cloglog family integrates the frailty out
//! (Gauss–Hermite quadrature, fixed latent SD = 1.0 on the formula path).
//!
//! Ground truth (independent of any gam β). The data are generated from the
//! latent-cloglog DGP at the SAME σ = 1.0 gam uses: a per-row frailty
//! `u_i ~ N(0,1)`, conditional probability `1 − exp(−exp(η_true(x_i) + u_i))`,
//! then `y_i ~ Bernoulli`. The marginal probability the model TARGETS,
//! `p_marg(x) = E_u[1 − exp(−exp(η_true(x) + u))]`, is the true generating
//! marginal and is the objective target. We evaluate it through gam's own
//! marginal-link evaluator at the KNOWN η_true (NOT at any fitted β), so it is a
//! genuine ground-truth curve — the link math, not an echo of gam's fit.
//!
//! Objective metrics asserted:
//!   1. PRIMARY (truth recovery): RMSE(gam_latent_prob, p_marg) is a small
//!      fraction of the marginal probability signal's range.
//!   2. FRAILTY GAIN (tool-free): the latent-cloglog fit recovers the true
//!      MARGINAL probability at least as well as a plain cloglog GAM on the
//!      IDENTICAL data (`rmse_latent ≤ rmse_plain_cloglog * 1.02`) — accounting
//!      for the frailty must not hurt, and on frailty-overdispersed data it
//!      should help. (A hard no-regression bound; a genuine tie passes.)
//!   3. EDF sanity and proper-mean range.
//!
//! No mature off-the-shelf tool fits this exact latent-cloglog marginal family
//! on arbitrary smooths, so — per the suite's reference-as-truth policy — the bar
//! is recovery of the analytically-known generating marginal, with the plain
//! cloglog GAM as the misspecified tool-free baseline.

use gam::families::inverse_link::apply_inverse_link_spec_vec;
use gam::estimate::FittedLinkState;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::types::{InverseLink, LatentCLogLogState, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

const N: usize = 500;
const LATENT_SD: f64 = 1.0; // the fixed latent SD the formula path uses.

/// True conditional linear predictor (cloglog scale).
fn eta_true(x: f64) -> f64 {
    0.8 * (PI * x).sin() - 0.7
}

#[test]
fn gam_latent_cloglog_recovers_frailty_marginal_probability() {
    init_parallelism();

    // The marginal-cloglog inverse link at the fixed σ = 1.0 — used both to build
    // the ground-truth marginal curve (at η_true) and to map gam's fitted η to
    // probabilities. Sharing gam's own evaluator makes the ground truth the exact
    // generating marginal, not a re-transcription.
    let latent_link =
        InverseLink::LatentCLogLog(LatentCLogLogState::new(LATENT_SD).expect("valid latent SD"));

    // ---- generate frailty-overdispersed binary data -------------------------
    let mut rng = StdRng::seed_from_u64(20260622);
    let ux = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let frailty = Normal::new(0.0, LATENT_SD).expect("gaussian frailty");
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = ux.sample(&mut rng);
        let ui = frailty.sample(&mut rng);
        // Conditional cloglog probability given the latent frailty draw.
        let eta_cond = eta_true(xi) + ui;
        let p_cond = 1.0 - (-(eta_cond.exp())).exp();
        let draw = if u01.sample(&mut rng) < p_cond { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(draw);
    }

    // The objective target: the TRUE MARGINAL probability p_marg(x), obtained by
    // integrating the frailty out of the cloglog at η_true (gam's exact marginal
    // evaluator at the known η — independent of any fit).
    let eta_true_vec: Vec<f64> = x.iter().map(|&xi| eta_true(xi)).collect();
    let p_marg = apply_inverse_link_spec_vec(&eta_true_vec, &latent_link)
        .expect("evaluate true marginal cloglog at eta_true");

    // The response must be a genuine two-class signal.
    let n_pos: usize = y.iter().filter(|&&v| v > 0.5).count();
    assert!(
        n_pos > 50 && n_pos < N - 50,
        "synthetic frailty-binary response is degenerate: {n_pos}/{N} positive"
    );
    let p_min = p_marg.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = p_marg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = p_max - p_min;
    assert!(
        signal_range > 0.25,
        "marginal probability signal too flat to test recovery: range={signal_range:.3}"
    );

    // ---- encode the shared dataset ------------------------------------------
    let headers: Vec<String> = ["x", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        rows.push(csv::StringRecord::from(vec![
            x[i].to_string(),
            y[i].to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode frailty-binary data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let p_cols = ds.headers.len();

    let eta_at_train = |resolvedspec: &gam::smooth::TermCollectionSpec,
                        beta: &ndarray::Array1<f64>|
     -> Vec<f64> {
        let mut grid = Array2::<f64>::zeros((N, p_cols));
        for i in 0..N {
            grid[[i, x_idx]] = x[i];
        }
        let design = build_term_collection_design(grid.view(), resolvedspec)
            .expect("rebuild s(x) design at training points");
        design.design.apply(beta).to_vec()
    };

    // ---- fit gam with the latent-cloglog binomial family --------------------
    let cfg_latent = FitConfig {
        family: Some("latent-cloglog-binomial".to_string()),
        ..FitConfig::default()
    };
    let result_latent = fit_from_formula("y ~ s(x)", &ds, &cfg_latent)
        .expect("gam latent-cloglog binomial smooth fit");
    let FitResult::Standard(fit_latent) = result_latent else {
        panic!("expected a standard GAM fit for latent-cloglog-binomial s(x)");
    };
    let latent_edf = fit_latent.fit.edf_total().expect("gam reports total edf (latent)");

    // The fitted family must actually be latent-cloglog.
    let latent_state_link = match &fit_latent.fit.fitted_link {
        FittedLinkState::LatentCLogLog { state } => InverseLink::LatentCLogLog(*state),
        other => panic!("expected a fitted latent-cloglog link state, got {other:?}"),
    };

    let gam_latent_eta = eta_at_train(&fit_latent.resolvedspec, &fit_latent.fit.beta);
    let gam_latent_prob = apply_inverse_link_spec_vec(&gam_latent_eta, &latent_state_link)
        .expect("evaluate fitted latent-cloglog marginal link on training etas");
    assert!(
        gam_latent_prob.iter().all(|&p| p > 0.0 && p < 1.0),
        "latent-cloglog marginal link must produce proper probabilities in (0,1)"
    );

    // ---- fit gam with a PLAIN cloglog link on the IDENTICAL data ------------
    // This baseline ignores the frailty: its η maps through the bare cloglog, so
    // it cannot represent the frailty-attenuated marginal mean.
    let cfg_plain = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("cloglog".to_string()),
        ..FitConfig::default()
    };
    let result_plain =
        fit_from_formula("y ~ s(x)", &ds, &cfg_plain).expect("gam plain cloglog smooth fit");
    let FitResult::Standard(fit_plain) = result_plain else {
        panic!("expected a standard GAM fit for binomial/cloglog s(x)");
    };
    let plain_link = InverseLink::Standard(StandardLink::CLogLog);
    let gam_plain_eta = eta_at_train(&fit_plain.resolvedspec, &fit_plain.fit.beta);
    let gam_plain_prob = apply_inverse_link_spec_vec(&gam_plain_eta, &plain_link)
        .expect("evaluate plain cloglog inverse link on training etas");

    // ---- OBJECTIVE: recover the true MARGINAL probability surface ------------
    let rmse_latent = rmse(&gam_latent_prob, &p_marg);
    let rmse_plain = rmse(&gam_plain_prob, &p_marg);

    eprintln!(
        "latent-cloglog frailty binary: n={N} signal_range={signal_range:.3} \
         latent_sd={LATENT_SD} latent_edf={latent_edf:.3} \
         RMSE_to_marginal_truth[latent={rmse_latent:.4} plain_cloglog={rmse_plain:.4}]"
    );

    // PRIMARY: the latent-cloglog fit recovers the true marginal probability.
    let truth_bar = 0.25 * signal_range;
    assert!(
        rmse_latent < truth_bar,
        "gam latent-cloglog failed to recover the true marginal probability curve: \
         RMSE_to_marginal_truth={rmse_latent:.4} (bound {truth_bar:.4} = 0.25*signal_range)"
    );

    // FRAILTY GAIN: accounting for the frailty must not regress vs the plain
    // cloglog fit on the marginal-truth metric; on frailty-overdispersed data it
    // should be at least as good. (2% slack for optimizer/round-off noise.)
    assert!(
        rmse_latent <= rmse_plain * 1.02,
        "latent-cloglog regressed vs plain cloglog at recovering the frailty marginal: \
         rmse_latent={rmse_latent:.4} > rmse_plain*1.02={:.4}",
        rmse_plain * 1.02
    );

    // EDF sanity: a real smooth was fitted.
    assert!(
        latent_edf > 1.0 && latent_edf < 25.0,
        "gam latent-cloglog effective dof out of sane range: {latent_edf:.3}"
    );
}
