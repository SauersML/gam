//! Issue #370 regression: the pre-fit identifiability audit must NOT crash on a
//! Bernoulli marginal-slope (BMS) model just because the log-slope channel has a
//! nonzero value at `beta = 0`.
//!
//! ## The bug
//!
//! The pre-fit identifiability audit builds each block's effective Jacobian once
//! at `beta = &[]` with `family_scalars = None` (see
//! `src/solver/identifiability_canonical.rs::BlockJacobianAsRowOp::from_callback`).
//! The BMS observed predictor is
//!
//! ```text
//!   η_i = q_i·c_i + s·g_i·z_i,   c_i = sqrt(1 + (s·g_i)²)
//!   q_i = M[i,:]·β_m + offset_m[i]      (marginal η)
//!   g_i = G[i,:]·β_s + offset_s[i]      (log-slope η)
//! ```
//!
//! where `offset_s = logslope_offset + logslope_baseline` and `logslope_baseline`
//! is a *data-driven pooled-probit pilot* value that is essentially never exactly
//! zero. The old block contract assumed "at β=0 every g_i == 0, so no
//! `BmsFamilyScalars` are needed" and hard-errored when it saw any nonzero g_i
//! while `family_scalars` was `None`. Because the fitted baseline (and any
//! user-supplied logslope offset) makes `g_i = offset_s[i] != 0` at β=0, that
//! guard fired for *every* BMS model — including the rigid `logslope_formula="1"`
//! control — making the entire score-warp / link-wiggle Python surface
//! unreachable.
//!
//! The fix (issue #367, commit `f32bb9b9`) makes both BMS blocks self-compute
//! `q_i, g_i, c_i, z_i` from owned data at the current β with NO caller-supplied
//! scalar contract. This test pins that fix at the *full-fit* level, which is the
//! angle issue #370 was filed from: it drives a complete `fit_model` BMS fit
//! (which runs the pre-fit audit end to end) while *forcing* a large nonzero
//! log-slope offset, so `g_i = offset_s[i]` is guaranteed far from zero at β=0 —
//! the exact precondition that used to abort the audit before any fitting began.
//!
//! Two variants mirror the issue's minimal repro:
//!   - rigid:  empty log-slope smooth + constant offset  (≈ `logslope_formula="1"`)
//!   - flex:   + score_warp / link_dev deviation blocks   (≈ `linkwiggle(...)`)

use gam::ResourcePolicy;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineKnotSpec, OneDimensionalBoundary,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const SEED: u64 = 0x370_0BAD_BA5E_11AE;

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Build a small `disease ~ s(bmi)` BMS problem (the issue's repro shape) with a
/// **forced large nonzero log-slope offset** so that `g_i = offset_s[i]` is far
/// from zero at β=0, independent of the data-driven pilot baseline.
fn build_problem(
    n: usize,
    flex: bool,
    logslope_offset_value: f64,
) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));

    // Column 0 is the smooth covariate (bmi-like), standardized to ~[-1, 1].
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut z = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);

    for i in 0..n {
        let bmi = rng.random_range(-1.0..1.0);
        // Latent score z ~ N(0,1) via Box-Muller would need pairing; a uniform
        // affine map is sufficient — z only enters η linearly through the slope.
        let zi = rng.random_range(-1.5..1.5);
        data[[i, 0]] = bmi;
        z[i] = zi;
        // η = smooth(bmi) signal + slope·z. Keep the slope modest so the latent
        // measure stays well behaved; the test is about the audit, not recovery.
        let eta = 0.8 * (std::f64::consts::PI * bmi).sin() - 0.4 + 0.3 * zi;
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let smooth = SmoothTermSpec {
        name: "f_bmi".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (-1.0, 1.0),
                    num_internal_knots: 6,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: BSplineBoundaryConditions::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth],
    };
    // Empty log-slope smooth == rigid `logslope_formula = "1"`: the log-slope
    // channel is driven purely by its offset, which we set large and nonzero.
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };

    let (score_warp, link_dev) = if flex {
        // internal_knots=6 mirrors the issue's `linkwiggle(internal_knots=6)`.
        let dev_cfg = DeviationBlockConfig::triple_penalty_default();
        (Some(dev_cfg.clone()), Some(dev_cfg))
    } else {
        (None, None)
    };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights: Array1::ones(n),
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset: Array1::zeros(n),
        // The crux of #370: a large nonzero log-slope offset guarantees
        // g_i = offset_s[i] = logslope_offset + logslope_baseline is far from
        // zero at β=0, the precondition that aborted the pre-fit audit.
        logslope_offset: Array1::from_elem(n, logslope_offset_value),
        frailty: FrailtySpec::None,
        score_warp,
        link_dev,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: None,
    };
    (data, spec)
}

fn run_fit(
    flex: bool,
    logslope_offset_value: f64,
) -> gam::families::bms::BernoulliMarginalSlopeFitResult {
    gam::init_parallelism();
    let (data, spec) = build_problem(400, flex, logslope_offset_value);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    match fit_model(request) {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("wrong FitResult variant"),
        // A panic here on the audit message is the #370 regression: the pre-fit
        // identifiability audit must not reject a nonzero-baseline BMS model.
        Err(e) => panic!(
            "issue #370 regression: BMS fit with a large nonzero log-slope offset \
             (flex={flex}, offset={logslope_offset_value}) failed before producing a \
             model — the pre-fit identifiability audit must self-compute the block \
             Jacobian scalars at β=0 rather than demanding family_scalars. error: {e}"
        ),
    }
}

/// Rigid control: `disease ~ s(bmi)` with `logslope_formula = "1"` and a large
/// nonzero log-slope offset. This is the exact case the issue says crashed for
/// *every* logslope formula, independent of linkwiggle.
#[test]
fn bms_rigid_nonzero_logslope_offset_audit_does_not_crash_370() {
    let out = run_fit(false, 1.7);
    // The fit must produce a finite, nonempty coefficient vector — i.e. it got
    // past the pre-fit audit and through the solver.
    assert!(
        !out.fit.beta.is_empty(),
        "issue #370: rigid BMS fit must produce coefficients, got an empty beta vector"
    );
    assert!(
        out.fit.beta.iter().all(|b| b.is_finite()),
        "issue #370: rigid BMS fit must produce finite coefficients, got {:?}",
        out.fit.beta
    );
}

/// Flex variant: + score_warp / link_dev deviation blocks (the `linkwiggle(...)`
/// surface from the issue), again with a large nonzero log-slope offset so the
/// audit sees g_i != 0 at β=0 across all three parametric/flex blocks.
#[test]
fn bms_flex_nonzero_logslope_offset_audit_does_not_crash_370() {
    let out = run_fit(true, 1.3);
    assert!(
        !out.fit.beta.is_empty(),
        "issue #370: flex BMS fit must produce coefficients, got an empty beta vector"
    );
    assert!(
        out.fit.beta.iter().all(|b| b.is_finite()),
        "issue #370: flex BMS fit must produce finite coefficients, got {:?}",
        out.fit.beta
    );
}
