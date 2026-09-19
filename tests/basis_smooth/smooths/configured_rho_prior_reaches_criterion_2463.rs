//! #2463: a configured `FitOptions::rho_prior` must reach the criterion the fit
//! minimizes — on EVERY smooth family, not just the ones the default-derivation
//! machinery happens to skip.
//!
//! `relax_smoothing_rho_prior` rewrites the caller's prior per coordinate before
//! the fit runs. On the relaxable families — B-spline (`ps`/`cr`/`bs`),
//! thin-plate, tensor-B-spline, pure Duchon — every coordinate the term owns was
//! overwritten unconditionally, so a configured prior was a silent no-op there:
//! the #2450 A/B/C measured a `ps` fit returning a BITWISE identical `ρ̂`, `edf`
//! and MISE under `Normal { mean: -6, sd: 0.25 }`, a prior pinning `λ` at `e⁻⁶`
//! to a quarter of a log unit. `matern` — which bails out of the rewrite for
//! length-alignment reasons — moved, which is what proved the difference was the
//! rewrite and not the data.
//!
//! That rewrite was right when it was written and is still right on its own
//! terms (#1266/#1271/#1867 each measured the old `Normal{0,3}` default harmful
//! for exactly those families). The defect was narrower: it could not tell
//! "caller left it unset" from "caller asked for this", because the default WAS
//! a configured-looking prior. #2450 made `RhoPrior::default()` `Flat`, so the
//! two became distinguishable and the rewrite can now relax the default without
//! discarding an instruction.
//!
//! Two halves of one contract are pinned here, because fixing the first by
//! disabling the derivation entirely would be a regression, not a fix:
//!
//!  1. a CONFIGURED prior reaches the criterion (`ρ̂` moves, and moves toward
//!     the prior);
//!  2. an UNSET prior still gets the derived default, in either of its two
//!     spellings — `Flat` and `GammaPrecision { shape: 1, rate: 0 }`, which this
//!     library documents as the same "unset" coordinate.

use gam::estimate::FitOptions;
use gam::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, fit_term_collection_forspec,
};
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    OneDimensionalBoundary,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

const N: usize = 200;

/// `y = sin(2πx) + N(0, 0.2²)` on a uniform design — a smooth with genuine
/// curvature, so REML lands `ρ̂` well away from either rail and the prior has
/// somewhere to pull it FROM.
fn dgp(seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut data = Array2::<f64>::zeros((N, 1));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let x = i as f64 / (N as f64 - 1.0);
        let u1 = next().max(1e-12);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        data[[i, 0]] = x;
        y[i] = (std::f64::consts::TAU * x).sin() + 0.2 * z;
    }
    (data, y)
}

/// A single-penalty cubic `ps` smooth: BSpline1D is the relaxable family, and
/// one penalty block means one `ρ` coordinate to read back.
fn bspline_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s_x".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 20,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            },
            shape: gam::terms::smooth::ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
    }
}

/// Fit under one prior and read back the converged `ρ̂₀ = log λ̂₀`.
fn rho_hat(data: &Array2<f64>, y: &Array1<f64>, prior: RhoPrior) -> f64 {
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let opts = FitOptions {
        rho_prior: prior.clone(),
        max_iter: 60,
        ..FitOptions::default()
    };
    let fitted = fit_term_collection_forspec(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &bspline_spec(),
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &opts,
    )
    .unwrap_or_else(|error| panic!("fit under {prior:?} must converge: {error:?}"));
    *fitted
        .fit
        .log_lambdas
        .first()
        .expect("single-penalty ps fit reports one log-lambda")
}

/// A prior no data could shrug off: it pins `λ` at `e⁻⁶` to a quarter of a log
/// unit, six e-folds below where REML puts this DGP. Any criterion that honours
/// its caller MUST move under it.
const ABSURD: RhoPrior = RhoPrior::Normal {
    mean: -6.0,
    sd: 0.25,
};

#[test]
fn configured_rho_prior_reaches_the_criterion_on_a_relaxable_smooth_2463() {
    let (data, y) = dgp(2463);
    let unset = rho_hat(&data, &y, RhoPrior::Flat);
    let configured = rho_hat(&data, &y, ABSURD);
    eprintln!("[2463] ps rho_hat: unset={unset:+.4} configured(N(-6,0.25))={configured:+.4}");

    // The discriminator from the #2450 A/B/C, stated as an assertion. Bitwise
    // equality here is the whole bug: it cannot mean "the prior is honoured and
    // happens not to matter", because a prior this tight six e-folds away
    // cannot fail to matter.
    assert!(
        unset.to_bits() != configured.to_bits(),
        "configured rho_prior never reached the criterion: rho_hat is bitwise \
         identical under RhoPrior::Flat and under {ABSURD:?} ({unset} vs {configured})"
    );
    // Direction, not just difference: a prior centred six e-folds BELOW the REML
    // optimum must pull `ρ̂` down, and with `sd = 0.25` it must dominate. A
    // one-unit margin is far inside the ≈13-unit gap the two arms actually show
    // while still refusing a fit that merely jitters.
    assert!(
        configured < unset - 1.0,
        "a Normal(-6, 0.25) prior must pull rho_hat DOWN from the REML optimum, \
         not merely perturb it: unset={unset}, configured={configured}"
    );
    // ...and it must land in the prior's own neighbourhood rather than somewhere
    // the criterion could have reached on its own.
    assert!(
        configured.is_finite() && configured < 0.0,
        "rho_hat under a prior pinning lambda at e^-6 must be negative, got {configured}"
    );
}

#[test]
fn both_spellings_of_an_unset_rho_prior_derive_the_same_default_2463() {
    let (data, y) = dgp(2463);
    let flat = rho_hat(&data, &y, RhoPrior::Flat);
    // `GammaPrecision { shape: 1, rate: 0 }` is "the explicit flat/default case"
    // by `RhoPrior`'s own documentation — cost, gradient and Hessian all vanish
    // under the MAP-in-λ convention. A gate written as `matches!(base, Flat)`
    // misclassifies it as configured and hands it back unrewritten, which would
    // silently drop the derived per-coordinate default (#1089 termination
    // widening, #1476 null-space degeneracy breaker) on a caller who asked for
    // nothing. This is the half of #2463 that a too-eager fix breaks.
    let gamma_flat = rho_hat(
        &data,
        &y,
        RhoPrior::GammaPrecision {
            shape: 1.0,
            rate: 0.0,
        },
    );
    eprintln!("[2463] ps rho_hat: Flat={flat:+.6} GammaPrecision{{1,0}}={gamma_flat:+.6}");
    assert_eq!(
        flat.to_bits(),
        gamma_flat.to_bits(),
        "the two spellings of an UNSET rho_prior must derive the same default \
         criterion: Flat gave {flat}, GammaPrecision{{shape:1, rate:0}} gave {gamma_flat}"
    );
}
