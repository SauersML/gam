//! #1074 regression guard for the STANDARD exponential-family REML outer loop
//! stalling on a Gamma 2-D tensor-product smooth.
//!
//! Root cause (verified): with the Gamma shape `k = 1/phi` ESTIMATED, the inner
//! solver re-derived `k` from each outer iterate's warm-start `eta` and froze it
//! only for the duration of that one inner solve. The Gamma working weight is
//! `W = prior*k` and the omitting-constants log-likelihood is `-ℓ = k*0.5*D`
//! (the `k`-dependent saturated normalizer is dropped, #359), so BOTH the
//! curvature `H = k*X'X + lambda*S` and the data-fit term `k*0.5*D` jumped with
//! every outer iterate's `k`. The realized REML cost surface developed
//! deterministic spikes between the smooth basin floors (a flat warm-start eta
//! at a just-rejected over-smoothed trial gives a small `k`, the fitted-surface
//! eta at the neighbor a ~2x larger one), so the analytic outer gradient — which
//! holds `k` fixed — could never match the cost's `k(rho)` motion. The projected
//! gradient floored well above tolerance and the ARC descent stalled on a
//! weakly-identified valley, railing `lambda` to the over-smoothed corner: a
//! te(x,z) k=7 Gamma fit recovered the true log-mean surface far worse than mgcv
//! (the #1074 te/Gamma tensor under-recovery).
//!
//! Fix: freeze `k` for the duration of the smoothing-parameter lambda search
//! (`GlmLikelihoodSpec::with_gamma_shape_frozen_for_search`, driven by the REML
//! state's `frozen_gamma_shape`), so `F(rho) = REML(rho, k_frozen)` is a
//! stationary function of rho and the loop converges. `k` is still ML-refreshed
//! at the single final, reported accept-fit. This mirrors the sibling
//! Negative-Binomial-theta (#1082) and Tweedie-phi (#1477) lambda-search freezes.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};
use std::f64::consts::PI;

const SHAPE: f64 = 4.0;

/// Build `(headers, rows)` for a Gamma/log dataset on the unit square drawn from
/// the smooth log-mean truth `eta(x,z) = 2 + sin(pi x) cos(pi z)`, columns
/// `x`, `z`, `y`. The constant offset keeps `mu = exp(eta)` comfortably away from
/// the origin so the Gamma deviance is well-conditioned.
fn synthetic_gamma_records(n: usize, seed: u64) -> (Vec<String>, Vec<StringRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let zi = u.sample(&mut rng);
            let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
            let mu = eta.exp();
            // Gamma(shape, scale = mu/shape) => E[y] = mu, Var = mu^2 / shape.
            let draw: f64 = Gamma::new(SHAPE, mu / SHAPE)
                .expect("gamma")
                .sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), zi.to_string(), draw.to_string()])
        })
        .collect();
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    (headers, rows)
}

/// The #1074 guard: an estimated-shape Gamma tensor fit must CONVERGE the outer
/// REML loop (the projected gradient must reach the outer tolerance) rather than
/// stalling on a shape-induced spiky valley and railing lambda to the
/// over-smoothed corner. Convergence is the knob-free signal that the
/// lambda-search shape freeze holds `F(rho)` stationary; the truth-recovery RMSE
/// bar itself is asserted by the `gam_tensor_te_2d_gamma_matches_mgcv` quality
/// test (which mgcv-matches to <0.1% after this fix).
#[test]
fn gamma_te_2d_outer_loop_converges_1074() {
    init_parallelism();
    let (headers, rows) = synthetic_gamma_records(300, 20260602);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");
    let cfg = FitConfig {
        family: Some("Gamma".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("gam gamma te fit");

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma te(x, z)");
    };

    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    eprintln!(
        "[#1074 guard] gamma te: outer_converged=certified grad_norm={:?} edf={:.3}",
        fit.fit.outer_gradient_norm, edf,
    );

    // Fit existence is the sealed convergence proof (SPEC 20).

    // The flexible REML optimum for this k=7 surface carries ~8 effective
    // degrees of freedom (mgcv reports edf ~= 8.1); a fit that stalled at the
    // over-smoothed corner collapses to edf ~= 4. Guard the basin, not a point.
    assert!(
        edf > 6.0,
        "#1074 regression: Gamma te recovered only edf={edf:.3} (expected ~8); \
         the optimizer railed lambda toward the over-smoothed corner",
    );
}

/// The #2361 guard: the λ-search shape freeze must capture ν at the solve's
/// CONVERGED η, so a plain single-smooth Gamma fit converges from a COLD start.
///
/// Root cause (#2361): the λ-search runs every inner solve with
/// `refine_dispersion_at_converged_eta = false` — deliberately, since
/// re-profiling ν against each trial λ's residuals would couple the scale to λ
/// and reward over-smoothing. #1074 then froze ν from the first converged
/// non-screening solve. But with the refresh off, the value that solve leaves on
/// `likelihood` is the shape measured at the half-converged warm-start η, whose
/// leftover spread in μ inflates the Gamma deviance and biases ν DOWN by >2×
/// (the same cold-start contamination `loop_driver`'s converged-η refresh loop
/// documents). Freezing THAT pinned the whole outer criterion to a
/// mis-specified dispersion.
///
/// Measured on this exact fit at c49433d6c, true ν = 4: a populated persistent
/// warm-start cache made the first inner solve converge in ONE PIRLS iteration
/// and froze ν = 3.96 (the fit converged); a cold cache made it take six and
/// froze ν = 1.22, after which the outer REML ground through all 200 iterations
/// and refused at |Pg| = 1.647e1 against a 1.193e-2 bound — an interior, PSD,
/// non-railed point. So the fit converged or not according to whether the
/// machine had fit this model before.
///
/// This guard runs the single-smooth Gamma fit and requires BOTH that the outer
/// loop certifies (fit existence is the sealed convergence proof, SPEC 20) and
/// that the recovered dispersion lands on the data's true conditional
/// dispersion rather than the >3× deflated cold-start value.
#[test]
fn gamma_single_smooth_outer_loop_converges_from_cold_start_2361() {
    init_parallelism();

    // n = 800 on the unit interval with the smooth log-mean truth used by the
    // Python surface smoke that surfaced this: eta(x) = 0.5 + 1.2 sin(2 pi x).
    let mut rng = StdRng::seed_from_u64(20260718);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..800)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let mu = (0.5 + 1.2 * (2.0 * PI * xi).sin()).exp();
            let draw: f64 = Gamma::new(SHAPE, mu / SHAPE)
                .expect("gamma")
                .sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), draw.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };

    // Pre-fix on a cold machine this call FAILS outright: the outer loop burns
    // its whole 200-iteration budget and returns RemlDidNotConverge.
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg)
        .expect("#2361: plain single-smooth Gamma fit must certify a stationary outer optimum");

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma s(x)");
    };

    let phi_hat = fit
        .fit
        .dispersion_phi()
        .expect("Gamma fit must retain a valid scalar response dispersion");
    let shape_hat = 1.0 / phi_hat;
    eprintln!(
        "[#2361 guard] gamma s(x): outer_converged=certified grad_norm={:?} shape_hat={shape_hat:.4}",
        fit.fit.outer_gradient_norm,
    );

    // The cold-start contamination deflated the shape to 1.22 against a truth of
    // 4.0. A factor-of-1.5 band absorbs the finite-sample noise of this draw
    // while still catching that >3x collapse.
    assert!(
        shape_hat > SHAPE / 1.5 && shape_hat < SHAPE * 1.5,
        "#2361 regression: recovered Gamma shape nu_hat = {shape_hat:.4} is far from the \
         true nu = {SHAPE:.1}; the lambda-search freeze captured a shape measured at a \
         half-converged eta instead of the converged one",
    );
}
