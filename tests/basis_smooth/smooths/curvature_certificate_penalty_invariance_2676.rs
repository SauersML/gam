//! #2676 end-to-end: what the curvature certificate is entitled to deflate, and
//! what it is not.
//!
//! The refusals this issue was filed for were:
//!
//! * `INDEFINITE CURVATURE AT INTERIOR OPTIMUM (curvature floor did not clear)`
//!   with `interior lambda_min = -5.048e-6` and `railed = []`, and
//! * `rho Hessian has negative curvature -1.452e-6 below the outer certificate's
//!   own resolution floor 1.444e-6 … a genuine contradiction rather than an
//!   unresolvable direction` — decided by a **0.55%** margin.
//!
//! On a direction where the criterion is EXACTLY constant in `lambda` the
//! rho-curvature is `Σ_k g_k t_k²` by the chain rule and nothing else, so the
//! gate compares a quantity against its own absolute value and the verdict is
//! the sign of a rounding residual. The repair is to deflate such a direction
//! rather than judge it. See `gam_solve::penalty_invariance`.
//!
//! # Why this file has two arms, and why the second one exists at all
//!
//! The premise "the `geo_disease_*_matern` penalty map carries an exact
//! redundancy" came from a screen that printed `cos` to six decimals, and
//! `1 - cos = delta²/2` — the cosine is the DEFECT SQUARED. Measured
//! (#2676 penalty-map probe, centers=10, n=1500, n_pcs=16):
//!
//! ```text
//!     length_scale      relative defect   certified nullity
//!       2.05e-2           2.079e-15              1
//!       1.64e-1 (Auto)    1.874e-5               0
//!       1.27e0  (fitted)  3.396e-1               0
//! ```
//!
//! The redundancy is a small-length-scale LIMIT of two genuinely different
//! operators, and it is not present at the geometry the `Auto` fit settles on.
//! So:
//!
//! * [`a_redundant_penalty_map_still_fits_and_certifies_2676`] gates the
//!   deflation on a geometry where the premise is TRUE, found by measurement
//!   rather than pinned to a constant;
//! * [`the_auto_geometry_carries_no_exact_invariance_and_still_certifies_2676`]
//!   pins the honest fact about the geometry the issue's own bench cells use —
//!   the fit certifies there and NOTHING is deflated — so the false premise
//!   cannot come back by inheritance.

use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionDesign, TermCollectionSpec,
};
use gam::solver::penalty_invariance::PenaltyMapInvariance;
use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu,
};
use gam::terms::construction::CanonicalPenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitRequest, FitResult, StandardFitRequest, StandardFitResult};
use ndarray::Array1;

// Chosen by sweeping the #2676 `geo_disease_matern` repro over
// `(centers, n, n_pcs)`: the smallest cell measured that BOTH reaches the
// #2676 code paths and completes (a few seconds in release). The exact
// redundancy is a property of the `(centers, n_pcs)` pair and of the geometry —
// `(10, 6)` has none at any scale, `(10, 16)` and `(24, 16)` have one below
// `~4e-2` — so neither constant is arbitrary and neither may be tuned to make a
// red test green.
const N_ROWS: usize = 1500;
const N_PCS: usize = 16;
const CENTERS: usize = 10;
const SEED: u64 = 20260226;

fn spec_at(length_scale: MaternLengthScale) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "geo".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..N_PCS).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::EqualMassCovarRepresentative {
                        num_centers: CENTERS,
                    },
                    periodic: None,
                    length_scale,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// `min_c ||S_j - c S_i||_F / ||S_i||_F` — the defect of the proportionality
/// claim, in the coordinate the penalty map's rank is actually decided in.
///
/// NOT `1 - cos`: that is this quantity SQUARED, and the squaring is what let a
/// `1.9e-5` pair print as `cos = 1.000000` for the whole life of this issue.
fn pair_defects(canonical: &[CanonicalPenalty]) -> Vec<(usize, usize, f64)> {
    let norms_sq: Vec<f64> = canonical
        .iter()
        .map(|penalty| penalty.local.iter().map(|value| value * value).sum::<f64>())
        .collect();
    let mut out = Vec::new();
    for i in 0..canonical.len() {
        for j in (i + 1)..canonical.len() {
            if canonical[i].col_range != canonical[j].col_range
                || norms_sq[i] == 0.0
                || norms_sq[j] == 0.0
            {
                continue;
            }
            let inner: f64 = canonical[i]
                .local
                .iter()
                .zip(canonical[j].local.iter())
                .map(|(a, b)| a * b)
                .sum();
            let scale = inner / norms_sq[i];
            let residual_sq: f64 = canonical[i]
                .local
                .iter()
                .zip(canonical[j].local.iter())
                .map(|(a, b)| {
                    let residual = b - scale * a;
                    residual * residual
                })
                .sum();
            out.push((i, j, (residual_sq / norms_sq[i]).sqrt()));
        }
    }
    out
}

fn canonicalize(design: &TermCollectionDesign) -> (Vec<CanonicalPenalty>, usize) {
    let specs: Vec<gam::terms::PenaltySpec> = design
        .penalties
        .iter()
        .map(|penalty| gam::terms::PenaltySpec::Block {
            local: penalty.local.clone(),
            col_range: penalty.col_range.clone(),
            prior_mean: penalty.prior_mean.clone(),
            structure_hint: penalty.structure_hint.clone(),
            op: penalty.op.clone(),
        })
        .collect();
    let p = design.design.ncols();
    let (canonical, _) = gam::terms::construction::canonicalize_penalty_specs(
        &specs,
        &vec![0usize; specs.len()],
        p,
        "#2676 acceptance",
    )
    .expect("canonicalization of a realized design must succeed");
    (canonical, p)
}

fn fit_at(length_scale: MaternLengthScale) -> Result<StandardFitResult, String> {
    fit_at_with(length_scale, SpatialLengthScaleOptimizationOptions::default())
}

fn fit_at_with(
    length_scale: MaternLengthScale,
    kappa_options: SpatialLengthScaleOptimizationOptions,
) -> Result<StandardFitResult, String> {
    let (x, y) = gam::test_support::synthetic::geo_disease_columns(N_ROWS, SEED);
    let n = y.len();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec: spec_at(length_scale),
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        estimate_tweedie_p: false,
        options: FitOptions {
            // Inference on, so the smoothing-correction inverse
            // (`invert_identified_rho_hessian`, this issue's path 2) runs
            // alongside the outer certificate (path 1).
            compute_inference: true,
            ..FitOptions::default()
        },
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    match result {
        Ok(FitResult::Standard(fit)) => Ok(fit),
        Ok(_) => panic!("the standard request must produce a standard result"),
        Err(error) => Err(error.to_string()),
    }
}

/// The other half, and the reason this file was rewritten: the geometry the
/// issue's own `geo_disease_*_matern` bench cells run at carries NO exact
/// invariance, the fit certifies there anyway, and nothing is deflated.
///
/// Three assertions, and the last two are what stop the false premise coming
/// back: the map certifies nothing, and the pair that used to be called
/// "structurally identical" is measurably distinct — reported as a defect, at a
/// stated multiple of the floor, so the next reader does not have to re-derive
/// it from a rounded cosine.
#[test]
fn the_auto_geometry_carries_no_exact_invariance_and_still_certifies_2676() {
    let fitted = fit_at(MaternLengthScale::auto())
        .unwrap_or_else(|error| panic!("the Auto-geometry Matern fit must not refuse: {error}"));
    assert!(
        fitted.fit.beta.iter().all(|value: &f64| value.is_finite()),
        "every coefficient of a certified fit must be finite"
    );

    let (canonical, p) = canonicalize(&fitted.design);
    let invariance = PenaltyMapInvariance::from_canonical_penalties(&canonical, p)
        .expect("the penalty-map factorization of a realized design must succeed");
    let defects = pair_defects(&canonical);
    let smallest = defects
        .iter()
        .map(|(_, _, defect)| *defect)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        invariance.dimension(),
        0,
        "the Auto geometry carries NO exact penalty-map invariance — if it now does, the \
         operator construction has changed and the first arm's premise has to be re-derived \
         rather than this assertion relaxed. defects={defects:?}"
    );
    assert!(
        smallest > invariance.resolution(),
        "and it must be measurably so: the closest pair sits at {smallest:.3e} against a defect \
         floor of {:.3e} ({:.3e}x). A defect at or under the floor with a certified nullity of \
         zero would mean the two disagree.",
        invariance.resolution(),
        smallest / invariance.resolution(),
    );
}
