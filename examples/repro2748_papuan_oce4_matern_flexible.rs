//! #2748 repro harness: the `papuan_oce4_matern_*` `_flexible` cells, which
//! fail on the OUTER certificate rather than on the inner frozen-index loop.
//!
//! # Why a second #2748 repro
//!
//! `repro2748_geo_disease_matern_flexible` covers the cluster's *inner* face —
//! `fit_binomial_mean_wiggle frozen-index fixed point did not converge`. Four
//! of the eight remaining sub-budget failures are a different terminal line:
//!
//! ```text
//! papuan_oce4_matern_k24: `binomial mean wiggle exact spatial hyper`
//!   NOT STATIONARY (|Pg|=3.277e0 > bound=2.169e-2) ... railed=[0,1,2,3,4,5]
//! papuan_oce4_matern_k6:  same, |Pg|=2.163e0 vs 2.171e-2, railed=[0,1,2,3,4,5]
//! papuan_oce4_matern_k12 / papuan_oce_matern_k12:
//!   custom-family outer not stationary, |Pg|=1.034e-1 / 3.938e-1 vs 4.800e-2
//! ```
//!
//! `railed=[0,…,5]` is EVERY coordinate of the joint θ = [ρ_η, ρ_wiggle, log κ]
//! vector, and `project_gradient_vector` keeps only the feasible-descent half of
//! a railed coordinate's gradient — so `|Pg| = 3.277` says the terminal point
//! has O(1) descent available INTO the box on coordinates the optimizer left
//! sitting on the boundary. Either the search never moved off its seed, or it
//! walked to a corner and stopped while descent remained.
//!
//! This binary reproduces that cell without a wheel, a venv or
//! `bench/run_suite.py`: `synthetic_papuan_oce_columns(6000, 20260315, 4)` — the
//! bench's own generator, called directly rather than inlined — one joint Matern
//! smooth over all four PCs with `length_scale=auto` (which is what turns on the
//! exact-joint `[ρ, ψ]` spatial route the failing message comes from), binomial
//! logit, and the `link(type=flexible(logit))` second stage.
//!
//! Run:
//! ```text
//! cargo run --release --example repro2748_papuan_oce4_matern_flexible -- \
//!     [k] [n] [n_pcs] [log_level] [flexible|fixed]
//! ```
//!
//! `fixed` is the built-in negative control: the identical fit with no link
//! wiggle, which is the lane that mints in CI.

use gam::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam::custom_family::BlockwiseFitOptions;
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::test_support::synthetic::papuan_oce_columns;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitRequest, FitResult, LinkWiggleConfig, StandardBinomialWiggleConfig, StandardFitRequest,
};
use ndarray::Array1;
use std::time::Instant;

fn smooth_term(n_pcs: usize, centers: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        frozen_parametric_residualization: None,
        name: "pcs".to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: (0..n_pcs).collect(),
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::EqualMassCovarRepresentative {
                    num_centers: centers,
                },
                periodic: None,
                // No `length_scale=` in the bench formula, so `Auto` — which is
                // what routes this fit through the exact-joint [rho, psi]
                // spatial optimizer whose certificate refuses.
                length_scale: gam::terms::basis::MaternLengthScale::auto(),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                // `rust_matern_decomposed_flexible` sets this; the
                // `_standard_flexible` companion leaves it false. Both lanes are
                // in the failing table, and the cheaper one is what runs here by
                // default (see `double_penalty` positional below if needed).
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

/// The link-wiggle request the `_flexible` lanes make — `materialize::standard`'s
/// construction for `link(type=flexible(logit))`.
fn wiggle_config() -> StandardBinomialWiggleConfig {
    let cfg = gam_spec::WigglePenaltyConfig::cubic_triple_operator_default();
    StandardBinomialWiggleConfig {
        link_kind: InverseLink::Standard(StandardLink::Logit),
        wiggle: LinkWiggleConfig {
            degree: cfg.degree,
            num_internal_knots: cfg.num_internal_knots,
            penalty_orders: cfg.penalty_orders.clone(),
            double_penalty: cfg.double_penalty,
        },
        refit_options: BlockwiseFitOptions {
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        },
    }
}

fn main() {
    // Log level is a positional argument, not `RUST_LOG`: reading the
    // environment is banned repo-wide (`build.rs`'s scanner).
    let args: Vec<String> = std::env::args().collect();
    let centers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let n_pcs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let level = args.get(4).map(String::as_str).unwrap_or("warn");
    let lane = args.get(5).map(String::as_str).unwrap_or("flexible");
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Warn);
    gam_solve::progress_log::set_log_level(level);

    eprintln!("[repro2748-papuan] lane={lane} centers={centers} n={n} n_pcs={n_pcs}");
    let (x, y) = papuan_oce_columns(n, 20260315, n_pcs);
    let n = y.len();
    let pos = y.iter().filter(|&&v| v > 0.5).count();
    eprintln!(
        "[repro2748-papuan] rows={n} cols={} prevalence={:.4}",
        x.ncols(),
        pos as f64 / n as f64
    );

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth_term(x.ncols(), centers)],
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        estimate_tweedie_p: false,
        options: fit_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        wiggle: match lane {
            "fixed" => None,
            _ => Some(wiggle_config()),
        },
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    let dt = t0.elapsed().as_secs_f64();

    match result {
        Ok(FitResult::Standard(s)) => {
            eprintln!(
                "[repro2748-papuan] OK in {dt:.2}s :: p={} finite={} warp_len={}",
                s.fit.beta.len(),
                s.fit.beta.iter().all(|v: &f64| v.is_finite()),
                s.wiggle_saved_warp_beta
                    .as_ref()
                    .map_or(0, |beta| beta.len()),
            );
        }
        Ok(_) => eprintln!("[repro2748-papuan] unexpected result kind in {dt:.2}s"),
        Err(e) => eprintln!("[repro2748-papuan] FAILED in {dt:.2}s :: {e}"),
    }
}
