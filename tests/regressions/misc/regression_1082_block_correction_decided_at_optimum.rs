//! #1082: a search decides the #784 block-local correction's admission at its
//! certified Laplace optimum, so the fitted model does not depend on where the
//! search started.
//!
//! The admission is frozen for the fit (#2748). It used to be decided at the
//! first evaluation whose skewness verdict engaged, which is a point of the
//! search's path, not of the model. On `gam_tensor_te_2d_poisson_matches_mgcv`
//! that point was the first evaluation (`max|γ| = 0.238` against `τ = 0.126`),
//! while the published optimum's verdict declines (`0.066`), so an `m = 8`
//! block was integrated at every later inner solution, about 142 s each on the
//! isotropic Smolyak route (job 1246493).
//!
//! Two small Poisson fits, each from starts `ρ = −6` and `ρ = 3`, measured
//! before the repair (job 1250999) and after it (job 1251000):
//!
//! - `y ~ s(x, k=8) + s(z, k=8)` on seed 7: the verdict engages at the start
//!   `−6` and declines at the optimum (`max|γ| = 0.083` against `τ = 0.173`).
//!   Before, the start `−6` latched the correction and the start `3` never did,
//!   so the two fits optimized different criteria. After, both decline at the
//!   optimum.
//! - `y ~ s(x, k=10)` on seed 1: the verdict engages at the optimum
//!   (`max|γ| = 0.286` against `τ = 0.200`). Before, both starts latched, but at
//!   different points of their paths (`−6` itself, and a mid-path point), so the
//!   latched rules and the published optima differed (`ρ₁` `4.803257` against
//!   `4.804257`). After, both admit at the Laplace optimum and latch the same
//!   rule and block.
//!
//! What the two starts share, and what they do not (#3280). The admission, the
//! latched rule and the latched block are properties of the model, so they are
//! the same from either start. The published `ρ` is not the same float: it is
//! the point where each start's search met its certificate, and nothing after a
//! declined decision moves it. The outer certificate resolves the criterion to
//! its statistical resolution `τ_stat = 1/(2n)` (`OuterProblemSize::
//! statistical_resolution`), a decrease that changes no reported quantity by
//! more than the error of the inference built on it, so two starts publish the
//! same model to that resolution, not to the bit. Measured at 7fff1617da (dy7
//! run w2thb-probe2): on seed 7 the starts end `4.8e-2` apart in `ρ₂` (a flat
//! direction, `λ₂ ≈ e^8.6`) with criterion values `6.6e-5` apart, against
//! `τ_stat = 3.1e-3`; on seed 1 they end `1.7e-10` apart in `ρ`, values `3e-14`
//! apart.

use csv::StringRecord;
use gam::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
use gam::estimate::{ExternalOptimOptions, optimize_external_designwith_heuristic_log_lambdas};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

struct PoissonFixture {
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    x: gam::matrix::DesignMatrix,
    penalties: Vec<gam::smooth::BlockwisePenalty>,
    opts: ExternalOptimOptions,
}

/// `n` rows, `log μ = log(base_mean) + amp·sin(2πx) + (amp/2)·cos(πz)`, Poisson
/// counts, drawn from `StdRng::seed_from_u64(seed)`.
fn poisson_fixture(seed: u64, n: usize, base_mean: f64, amp: f64, formula: &str) -> PoissonFixture {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = unit.sample(&mut rng);
        let z = unit.sample(&mut rng);
        let eta = base_mean.ln() + amp * (2.0 * PI * x).sin() + 0.5 * amp * (PI * z).cos();
        let count: f64 = Poisson::new(eta.exp())
            .expect("a finite positive Poisson rate")
            .sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            z.to_string(),
            count.to_string(),
        ]));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture");
    let config = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let model = gam::materialize(formula, &data, &config).expect("materialize the fixture");
    let FitRequest::Standard(request) = model.request else {
        panic!("a Poisson smooth is a standard request");
    };
    let design = build_term_collection_design(request.data.view(), &request.spec)
        .expect("build the fixture design");
    PoissonFixture {
        y: (*request.y).clone(),
        weights: (*request.weights).clone(),
        offset: &*request.offset + &design.affine_offset,
        x: design.design.clone(),
        penalties: design.penalties.clone(),
        opts: ExternalOptimOptions {
            family: LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: true,
            max_iter: 300,
            tol: 1.0e-8,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            firth_bias_reduction: Some(false),
            rho_prior: Default::default(),
            persistent_warm_start_store: None,
        },
    }
}

/// What a fit published: its smoothing parameters and criterion value, and
/// whether the #784 correction is part of its criterion, read from the audit of
/// its last outer evaluation, with the latched rule and block when it is.
struct Published {
    log_lambdas: Vec<f64>,
    criterion: f64,
    engaged: bool,
    axis_orders: Vec<usize>,
    block_cols: Vec<usize>,
}

fn fit_from(fixture: &PoissonFixture, start: f64) -> Published {
    let start = vec![start; fixture.penalties.len()];
    enable_rho_outer_audit();
    let fit = optimize_external_designwith_heuristic_log_lambdas(
        fixture.y.view(),
        fixture.weights.view(),
        fixture.x.clone(),
        fixture.offset.view(),
        fixture.penalties.clone(),
        Some(&start),
        &fixture.opts,
    )
    .expect("the fixture fits");
    let audit = take_rho_outer_audit().expect("the fit's outer evaluations ran on this thread");
    assert!(fit.outer_converged, "the fixture's outer search certifies");
    let (axis_orders, block_cols) = audit
        .quadrature_marginal
        .as_ref()
        .map(|record| (record.axis_orders.clone(), record.block_cols.clone()))
        .unwrap_or_default();
    Published {
        log_lambdas: fit.log_lambdas.to_vec(),
        criterion: fit
            .reml_score
            .expect("a certified Poisson fit publishes its criterion value"),
        engaged: audit.quadrature_marginal_engaged,
        axis_orders,
        block_cols,
    }
}

fn listed(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.12e}"))
        .collect::<Vec<String>>()
        .join(", ")
}

#[test]
fn block_correction_admission_does_not_depend_on_the_start_1082() {
    // Registers the Laplace marginal corrector; without it the correction
    // declines before the diagnostic and both branches below are vacuous.
    init_parallelism();
    for (seed, n, base_mean, amp, formula, admitted) in [
        (7_u64, 160_usize, 0.7, 1.0, "y ~ s(x, k=8) + s(z, k=8)", false),
        (1, 120, 0.6, 1.2, "y ~ s(x, k=10)", true),
    ] {
        let fixture = poisson_fixture(seed, n, base_mean, amp, formula);
        let low = fit_from(&fixture, -6.0);
        let high = fit_from(&fixture, 3.0);
        // The criterion's statistical resolution over the `n` rows the route
        // declares (`OuterProblemSize::statistical_resolution`).
        let tau_stat = 0.5 / fixture.y.len() as f64;
        let criterion_gap = (low.criterion - high.criterion).abs();
        eprintln!(
            "[#1082 pin] seed={seed} `{formula}`: start -6 engaged={} criterion={:.15e} \
             log_lambdas=[{}]; start 3 engaged={} criterion={:.15e} log_lambdas=[{}]; \
             criterion gap {criterion_gap:.3e} against tau_stat {tau_stat:.3e}",
            low.engaged,
            low.criterion,
            listed(&low.log_lambdas),
            high.engaged,
            high.criterion,
            listed(&high.log_lambdas)
        );
        // Each fixture exercises its branch: a decision that does not hold its
        // expected value would make the agreement below a comparison of two
        // equally wrong fits.
        assert_eq!(
            high.engaged, admitted,
            "#1082: seed {seed} `{formula}` from start 3 must be {} at its Laplace optimum",
            if admitted { "admitted" } else { "declined" }
        );
        assert_eq!(
            low.engaged, high.engaged,
            "#1082: seed {seed} `{formula}`: the start -6, where the skewness verdict engages, \
             decided the correction's admission differently from the start 3. The admission is \
             a property of the model, decided at its Laplace optimum, not of the search's path."
        );
        assert_eq!(
            (&low.axis_orders, &low.block_cols),
            (&high.axis_orders, &high.block_cols),
            "#1082: seed {seed} `{formula}`: the two starts latched different rules or blocks"
        );
        // The two fits certify optima of one criterion, so they must publish
        // the same model to the resolution their certificates decide at. A
        // Newton-decrement verdict certifies a point only where the decrease
        // left to the minimum, rounding included, is within
        // `max(τ_stat − band_f, band_f) ≤ τ_stat`, where `band_f` is the
        // criterion value's own rounding band: no verdict is taken where
        // `band_f > τ_stat` (`outer_decrement_bands`), and `V − V* ≤ λ²` is
        // the self-concordant bound the verdict rests on. Each published value
        // is within `band_f ≤ τ_stat` of its exact value, so both lie in
        // `[V* − τ_stat, V* + 2·τ_stat]` and differ by at most `3·τ_stat`. The
        // #784-corrected criterion has no analytic ρ-Hessian (#3139), so its
        // fits certify on the per-coordinate gradient band instead, and there
        // the same bar is this pin's requirement that both starts publish one
        // answer to the criterion's resolution. Optima of two criteria, or two
        // local optima, whose values differ by more than that fail it.
        assert!(
            criterion_gap <= 3.0 * tau_stat,
            "#1082: seed {seed} `{formula}`: the two starts published fits whose criterion \
             values differ by {criterion_gap:.6e} ({:.15e} against {:.15e}), beyond \
             3·tau_stat = {:.6e}, the most two certified optima of one criterion can differ \
             by (log_lambdas [{}] against [{}])",
            low.criterion,
            high.criterion,
            3.0 * tau_stat,
            listed(&low.log_lambdas),
            listed(&high.log_lambdas)
        );
    }
}
