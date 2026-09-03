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
//! Project each spatial term's `length_scale` onto the caller's window ONCE, in
//! the spec, before the baseline fit — so the incumbent and the joint seed are
//! the same point by construction and the caller's `min_length_scale` stays
//! authoritative. (The alternative — widening the ψ box to contain the raw
//! incumbent — would admit a length scale BELOW the caller's own bound.)
//!
//! ## What this file asserts, and what it deliberately does not
//!
//! It asserts the fixture really is the out-of-window arm (the projection moves
//! it, so the test cannot pass vacuously) and that the fit no longer refuses on
//! either θ₀-identity ground. It does NOT assert the fit succeeds: the in-window
//! arm (`length_scale = 1e-2`, which is where this one now lands) has a
//! SEPARATE, pre-existing outer non-stationarity failure — `|Pg| = 5.346`
//! against `bound = 3.889e-2` after the outer budget — that #2726 never touched
//! and that this repair must not be credited with fixing.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec, get_spatial_length_scale, project_spatial_length_scales_in_spec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};

/// The spec's `length_scale`, below the caller's window on purpose.
const RAW_LENGTH_SCALE: f64 = 1.0e-3;
const MIN_LENGTH_SCALE: f64 = 1.0e-2;
const MAX_LENGTH_SCALE: f64 = 1.0e2;
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
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn kappa_options(max_outer_iter: usize) -> SpatialLengthScaleOptimizationOptions {
    SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: MIN_LENGTH_SCALE,
        max_length_scale: MAX_LENGTH_SCALE,
        pilot_subsample_threshold: 0,
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
        kronecker_penalty_system: None,
        kronecker_factored: None,
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
        estimate_tweedie_p: false,
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

/// Non-vacuity control, run first: the fixture MUST be the out-of-window arm.
/// If the spec's `length_scale` were already inside the caller's window, the
/// projection would be inert and the gate below would pass without exercising
/// anything.
#[test]
fn regression_2726_fixture_is_the_out_of_window_arm() {
    let opts = kappa_options(15);
    let mut spec = spec_1d(RAW_LENGTH_SCALE);
    let moved = project_spatial_length_scales_in_spec(&mut spec, &[0], &opts)
        .expect("duchon term accepts a projected length scale");
    assert_eq!(
        moved,
        vec![(0usize, RAW_LENGTH_SCALE, MIN_LENGTH_SCALE)],
        "the #2726 fixture must be the arm the caller's window excludes"
    );
    assert_eq!(
        get_spatial_length_scale(&spec, 0),
        Some(MIN_LENGTH_SCALE),
        "the projection is written back into the spec, so the baseline fit and \
         the joint seed read the same value"
    );
    // The step the two routes used to be apart by, printed rather than
    // re-asserted: `crates/gam-terms/.../term_specs.rs` pins it bit-exactly.
    println!(
        "[2726] projection step in psi = {:.17e}",
        MIN_LENGTH_SCALE.ln() - RAW_LENGTH_SCALE.ln()
    );
}

/// The gate. The refusal was budget-independent (bit-identical `gap` at 15 and
/// 60 outer iterations), so both budgets are exercised: a repair that merely
/// moved the failure past a budget would still trip the smaller one.
#[test]
fn regression_2726_joint_and_scalar_rho_routes_share_theta0() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
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
/// this out-of-window fixture PRODUCES A FIT. That is strictly stronger — it
/// survives any subsequent change to how the cross-route comparison is
/// denominated — and it is the thing a user of `min_length_scale` cares about.
///
/// It is also the arm the #2760 repairs are about: at `n = 600` the scalar-ρ
/// incumbent already puts 4 of 5 coordinates below `−JOINT_RHO_BOUND`, so the
/// pre-#2760 box pasted its lower wall onto them and the certified n-free
/// ψ-Gram surrogate was still the measure at certification time. Both budgets
/// are exercised for the same reason the sibling gate exercises both: the
/// original refusal was budget-independent.
#[test]
fn regression_2726_out_of_window_fixture_still_fits() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    for max_outer_iter in [15usize, 60] {
        let outcome = run_fit(max_outer_iter);
        assert!(
            outcome.is_ok(),
            "the out-of-window length-scale fixture must FIT at \
             max_outer_iter={max_outer_iter}; it refused with: {}",
            outcome.unwrap_err(),
        );
    }
}
