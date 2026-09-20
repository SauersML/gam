//! #2726: the joint [ρ, ψ] spatial route and the scalar-ρ route disagreed about
//! the criterion AT THE SAME θ₀ — a premise the refusal stated in prose while it
//! was false.
//!
//! ## What was measured
//!
//! On the 1-D hybrid-Duchon fixture below with `length_scale = 1e-3` against a
//! caller window of `[1e-2, 1e2]`, the monotonicity certificate in
//! `try_exact_joint_spatial_length_scale_optimization` refused with
//!
//! ```text
//! joint_seed = -3.87252440418254218e3
//! baseline   = -3.87252440418247033e3   -- WRONG: 98.857 apart before the fix
//! gap        =  9.88570261368004140e1   accept_tol = 3.87252440418247050e-5
//! ```
//!
//! a factor of `2.55e6` over tolerance and bit-identical at `max_outer_iter` 15
//! and 60, i.e. budget-independent. The cause was not a criterion defect and not
//! a solver failure: the ψ seed was built from `length_scale` PROJECTED onto the
//! caller's window while the scalar-ρ incumbent it was graded against was fit
//! from the spec's RAW value, so the two routes evaluated one criterion at two
//! points `ln 10` apart. Restoring a shared θ₀ collapsed the gap to `7.185e-11`
//! (`1.376e12`×, `1.855e-14` relative — roundoff on a criterion of magnitude
//! `3.87e3`).
//!
//! ## What the repair is
//!
//! The repair first projected each term's `length_scale` onto the caller's
//! window once, in the spec, before the baseline fit. #2902 then deleted the
//! caller window itself (SPEC rule 20): the κ search box is derived from the
//! term's point cloud, so there is no projection left, and the joint seed is −ln
//! of the incumbent's own scale by construction.
//!
//! ## What this file asserts
//!
//! That the raw-`length_scale` fixture never refuses on either θ₀-identity
//! ground, and that it fits. The names still say "out of window": this is the
//! fixture that sat below the old `[1e-2, 1e2]` caller window.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};

/// The spec's `length_scale`, the value that sat below the old caller window.
const RAW_LENGTH_SCALE: f64 = 1.0e-3;
const N_ROWS: usize = 600;

/// Fragments of the refusals that can only fire when θ₀ is not shared between
/// the joint and the scalar-ρ route. Either one appearing means #2726 has
/// regressed; every other refusal is a different defect.
///
/// gam#2760 note on the FIRST string. It is the #2454 cross-route criterion
/// comparison, and that comparison is now a warning rather than a refusal —
/// two independent assemblies of one criterion have an `O(ε·κ)` forward error
/// and no fixed relative constant can denominate it (see
/// `try_exact_joint_spatial_length_scale_optimization`). The string is kept
/// because it costs nothing and a future revision could reinstate it, but the
/// load-bearing half of this gate is now the SECOND string — #2726's own ψ
/// guard, which is untouched and still a hard refusal — together with
/// `regression_2726_out_of_window_fixture_still_fits` below, which asserts the
/// stronger property the original refusal was a proxy for.
const THETA0_REFUSALS: [&str; 2] = [
    "disagree about the criterion AT THE SAME POINT theta0",
    "the scalar-rho incumbent was never realized at",
];

fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        y[i] = t.sin();
    }
    (x, y)
}

fn spec_1d(length_scale: f64) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(length_scale),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec {
                        mass: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        tension: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        stiffness: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                    },
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    }
}

fn kappa_options(max_outer_iter: usize) -> SpatialLengthScaleOptimizationOptions {
    SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter,
        rel_tol: 1e-5,
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
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-6,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

/// `Ok(())` when the fit succeeded, `Err(message)` with the refusal text.
fn run_fit(max_outer_iter: usize) -> Result<(), String> {
    let (x, y) = simulate_1d_gaussian(N_ROWS);
    let weights = Array1::ones(N_ROWS);
    let offset = Array1::zeros(N_ROWS);
    let outcome = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(weights),
        offset: std::sync::Arc::new(offset),
        spec: spec_1d(RAW_LENGTH_SCALE),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(),
        kappa_options: kappa_options(max_outer_iter),
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    match outcome {
        Ok(FitResult::Standard(_)) => Ok(()),
        // `FitResult` does not implement `Debug`, so name the variant rather than
        // formatting it -- `{other:?}` did not compile and turned every target in
        // the workspace red (the root `build.rs` aborts, so no crate-scoped build
        // could see it).
        Ok(_) => Err("unexpected fit result variant (expected FitResult::Standard)".to_string()),
        Err(e) => Err(format!("{e}")),
    }
}

/// The gate. The refusal was budget-independent (bit-identical `gap` at 15 and
/// 60 outer iterations), so both budgets are exercised: a repair that merely
/// moved the failure past a budget would still trip the smaller one.
#[test]
fn regression_2726_joint_and_scalar_rho_routes_share_theta0() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Debug);
    for max_outer_iter in [15usize, 60] {
        let outcome = run_fit(max_outer_iter);
        match &outcome {
            Ok(()) => println!("[2726] max_outer_iter={max_outer_iter}: fit OK"),
            Err(message) => {
                println!("[2726] max_outer_iter={max_outer_iter}: fit refused: {message}");
                for refusal in THETA0_REFUSALS {
                    assert!(
                        !message.contains(refusal),
                        "#2726 regressed at max_outer_iter={max_outer_iter}: the joint \
                         [rho, psi] route is being graded against an incumbent it does not \
                         share theta_0 with.\nrefusal: {message}"
                    );
                }
            }
        }
    }
}

/// The property the #2454 refusal was a proxy for, asserted directly (gam#2760).
///
/// "The two routes do not disagree about the criterion at θ₀" is a statement
/// about two floating-point assemblies; what it was standing in for is that
/// this raw-scale fixture PRODUCES A FIT. That is strictly stronger — it
/// survives any subsequent change to how the cross-route comparison is
/// denominated — and it is the thing a user of an explicit `length_scale` cares
/// about.
///
/// It is also the arm the #2760 repairs are about: at `n = 600` the scalar-ρ
/// incumbent already puts 4 of 5 coordinates below `−JOINT_RHO_BOUND`, so the
/// pre-#2760 box pasted its lower wall onto them and the certified n-free
/// ψ-Gram surrogate was still the measure at certification time. Both budgets
/// are exercised for the same reason the sibling gate exercises both: the
/// original refusal was budget-independent.
#[test]
fn regression_2726_out_of_window_fixture_still_fits() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Debug);
    // Two budgets: the fixture's short one, on which a stop is a statement
    // about the budget and not about the criterion, and the engine's own. The
    // #2726 defect refused identically on both; a budget stop on the short arm
    // is not that defect, and the fit on the engine's budget is the claim.
    let production = SpatialLengthScaleOptimizationOptions::default().max_outer_iter;
    for max_outer_iter in [15usize, production] {
        let outcome = run_fit(max_outer_iter);
        match outcome {
            Ok(()) => {}
            Err(reason) if max_outer_iter < production && reason.contains("iteration_budget") => {
                eprintln!(
                    "[2726] the short-budget arm ({max_outer_iter} iterations) stopped on its \
                     budget, which is not the #2726 refusal: {reason}"
                );
            }
            Err(reason) => panic!(
                "the out-of-window length-scale fixture must FIT at max_outer_iter={max_outer_iter}; \
                 it refused with: {reason}"
            ),
        }
    }
}
