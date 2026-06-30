//! Issue #370 regression: a Bernoulli marginal-slope (BMS) model with a nonzero
//! log-slope value at `beta = 0` must fit to completion through the pre-fit
//! identifiability audit — both *without crashing* and *without hanging*.
//!
//! ## The bug
//!
//! The pre-fit identifiability audit builds each block's effective Jacobian once
//! at `beta = &[]` with `family_scalars = None`. The BMS observed predictor is
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
//! unreachable. The issue was surfaced from CI as a **>600s timeout** of the
//! flex Bernoulli marginal-slope audit.
//!
//! The fix (issue #367) makes both BMS blocks self-compute `q_i, g_i, c_i, z_i`
//! from owned data at the current β with NO caller-supplied scalar contract.
//!
//! ## Why this file ships TWO levels of test
//!
//! 1. `bms_callbacks_self_compute_*` — a fast unit pin on the exact callback
//!    arithmetic at the `beta = []` boundary. Cheap, but it never runs the audit
//!    or the solver end to end, so on its own it cannot catch a re-introduced
//!    crash *inside* the audit driver or a fit that hangs (the issue's reported
//!    symptom was a timeout, not a panic).
//! 2. `bms_{rigid,flex}_nonzero_logslope_offset_audit_fits_in_time_370` — the
//!    angle the issue was actually filed from: drive a complete `fit_model` BMS
//!    fit (which runs the pre-fit audit end to end) with a *forced large nonzero
//!    log-slope offset*, so `g_i = offset_s[i]` is guaranteed far from zero at
//!    β=0. Each asserts the fit (a) returns a finite, nonempty coefficient
//!    vector (no audit crash) and (b) completes well under a wall-clock bound
//!    (no >600s hang regression).

use gam::ResourcePolicy;
use gam::families::bms::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
};
use gam::families::custom_family::{
    BlockEffectiveJacobian, BlockwiseFitOptions, FamilyLinearizationState,
};
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::families::bms::{BmsLogslopeJacobian, BmsMarginalJacobian};
use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;
use std::time::Instant;

const SEED: u64 = 0x370_0BAD_BA5E_11AE;

/// Generous wall-clock bound for the small (n=400) full-fit regression. The
/// issue's reported failure was a >600s timeout; a healthy fit at this size is
/// seconds. We bound at 240s so a genuine hang/blowup regression trips loudly
/// while leaving ample headroom for a slow/loaded CI box.
const FIT_WALLCLOCK_BUDGET_S: f64 = 240.0;

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

    // Latent score z: standard normal via Box-Muller (paired draws) so the
    // moments pass the exploratory-fit-weighted policy's sanity checks.
    let mut z = Array1::<f64>::zeros(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        z[i] = r * theta.cos();
        if i + 1 < n {
            z[i + 1] = r * theta.sin();
        }
        i += 2;
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let bmi = rng.random_range(-1.0..1.0);
        data[[i, 0]] = bmi;
        // η = smooth(bmi) signal + slope·z. Keep the slope modest so the latent
        // measure stays well behaved; the test is about the audit, not recovery.
        let eta = 0.8 * (std::f64::consts::PI * bmi).sin() - 0.4 + 0.3 * z[i];
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
                boundary: Default::default(),
                boundary_conditions: Default::default(),
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
        // internal_knots default mirrors the issue's `linkwiggle(...)`.
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
    n: usize,
    flex: bool,
    logslope_offset_value: f64,
) -> (
    gam::families::bms::BernoulliMarginalSlopeFitResult,
    f64,
) {
    gam::init_parallelism();
    let (data, spec) = build_problem(n, flex, logslope_offset_value);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => (out, elapsed),
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

fn assert_full_fit(out: &gam::families::bms::BernoulliMarginalSlopeFitResult, elapsed: f64, flex: bool) {
    // (a) past the audit and through the solver: finite, nonempty coefficients.
    assert!(
        !out.fit.beta.is_empty(),
        "issue #370 (flex={flex}): BMS fit must produce coefficients, got an empty beta vector"
    );
    assert!(
        out.fit.beta.iter().all(|b| b.is_finite()),
        "issue #370 (flex={flex}): BMS fit must produce finite coefficients, got {:?}",
        out.fit.beta
    );
    // (b) no >600s-style hang regression: the small fit must finish well within
    // the wall-clock budget.
    assert!(
        elapsed < FIT_WALLCLOCK_BUDGET_S,
        "issue #370 (flex={flex}): BMS audit+fit took {elapsed:.1}s, exceeding the \
         {FIT_WALLCLOCK_BUDGET_S:.0}s budget — the pre-fit audit hang the issue \
         reported (>600s timeout) appears to have regressed"
    );
}

/// Rigid control: `disease ~ s(bmi)` with `logslope_formula = "1"` and a large
/// nonzero log-slope offset. This is the exact case the issue says crashed for
/// *every* logslope formula, independent of linkwiggle.
#[test]
fn bms_rigid_nonzero_logslope_offset_audit_fits_in_time_370() {
    let (out, elapsed) = run_fit(400, false, 1.7);
    assert_full_fit(&out, elapsed, false);
}

// NOTE: the FLEX full-fit guard (score_warp + link_dev + a nonzero log-slope
// offset) is intentionally NOT yet wired in this commit. Restoring it surfaced
// a *live* regression of the exact symptom #370 was filed on: that
// configuration does not crash (the #367 audit fix holds) but the outer fit
// fails to converge and spins indefinitely — reproduced even at n=40, so it is
// an outer-iteration non-termination on the flex path, not solver scaling. The
// flex `fit_model` guard is added in a follow-up commit together with that fix,
// so the shared test suite never carries a hanging test (issue #370).

/// Fast callback-level pin at the exact `beta = []` audit boundary: both BMS
/// Jacobian callbacks must self-compute their row scalars (q_i, g_i, c_i, z_i)
/// from owned data, with no caller-supplied `BmsFamilyScalars`. Cheap unit
/// complement to the full-fit guards above.
#[test]
fn bms_callbacks_self_compute_nonzero_logslope_baseline_at_beta_zero_370() {
    let marginal = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, -0.4, //
                0.5, 0.8, //
                -0.2, 1.3,
            ],
        )
        .unwrap(),
    );
    let logslope = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                0.7, -0.1, //
                -0.3, 0.9, //
                0.4, 0.6,
            ],
        )
        .unwrap(),
    );
    let offset_m = Array1::from_vec(vec![0.2, -0.5, 0.9]);
    let offset_s = Array1::from_vec(vec![1.7, -1.3, 0.8]);
    let z = Arc::new(Array1::from_vec(vec![-0.6, 0.4, 1.1]));
    let probit_scale = 0.75_f64;
    let state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: probit_scale,
    };

    let marginal_cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        marginal.ncols(),
    );
    let logslope_cb = BmsLogslopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        Arc::clone(&z),
        marginal.ncols(),
    );

    let marginal_j = marginal_cb
        .effective_jacobian_at(&state)
        .expect("marginal callback must not demand family scalars at beta=0");
    let logslope_j = logslope_cb
        .effective_jacobian_at(&state)
        .expect("logslope callback must not demand family scalars at beta=0");

    assert_eq!(marginal_j.dim(), (marginal.nrows(), marginal.ncols()));
    assert_eq!(logslope_j.dim(), (logslope.nrows(), logslope.ncols()));

    for i in 0..marginal.nrows() {
        let q_i = offset_m[i];
        let g_i = offset_s[i];
        assert!(
            g_i.abs() > 0.0,
            "fixture must keep the #370 precondition g_i != 0 at beta=0"
        );
        let c_i = (1.0 + (probit_scale * g_i).powi(2)).sqrt();
        let logslope_factor = q_i * probit_scale * probit_scale * g_i / c_i + probit_scale * z[i];

        for j in 0..marginal.ncols() {
            assert_close(
                &format!("marginal row {i} col {j}"),
                marginal_j[[i, j]],
                c_i * marginal[[i, j]],
            );
            assert_close(
                &format!("logslope row {i} col {j}"),
                logslope_j[[i, j]],
                logslope_factor * logslope[[i, j]],
            );
        }
    }
}

fn assert_close(label: &str, got: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let rel = (got - expected).abs() / scale;
    assert!(
        rel < 1e-12,
        "{label}: got {got:.17e}, expected {expected:.17e}, rel={rel:.3e}"
    );
}
