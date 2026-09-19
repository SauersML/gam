#![cfg(test)]
//! Finite-difference gates for every survival marginal-slope ψ lane
//! (#2765 / #2767, the `#979`/`#1040` lane).
//!
//! The family publishes three objects per outer ψ coordinate, and the generic
//! outer-REML hyper assembly (`build_psi_hyper_coords`) consumes all three:
//!
//! ```text
//!   objective_psi = ∂_ψ  ℓ̄ |_β
//!   score_psi     = ∂_ψ ∇_β ℓ̄ |_β
//!   hessian_psi   = ∂_ψ ∇²_β ℓ̄ |_β
//! ```
//!
//! where `ℓ̄` is the family's own joint objective — the one
//! `exact_newton_joint_gradient_evaluation` and `exact_newton_joint_hessian`
//! report — and there are two structurally different kinds of ψ:
//!
//! * a **design** axis, which moves one block's design (`X(ψ) = X + ψ·X_ψ`);
//!   this is the matern/duchon length scale, served by `psi_terms_inner`;
//! * a **baseline** axis, which moves the parametric baseline chart's own
//!   coordinates and therefore the three offset channels of the location
//!   index; served by `baseline_exact_joint_psi_terms_with_options`.
//!
//! Nothing compared any of them against a finite difference of the functions
//! they claim to differentiate. The shipped ψ coverage checks FINITENESS,
//! subsample-vs-unsampled equality, and batched-vs-per-axis agreement — every
//! one of which a *consistently wrong* derivative passes. Meanwhile the
//! end-to-end audit
//! (`tests/survival/survival/survival_marginal_slope_outer_gradient_fd_1040.rs`)
//! records this criterion's ψ gradient disagreeing with its own Ridders oracle
//! by `1.000e0` and `1.377e-1` relative, with the oracle's uncertainty six to
//! seven orders below the gap, and #2765's acceptance fit dies in a line search
//! along a pure baseline-ψ direction. Neither gap had anywhere to be
//! attributed, because no gate sat between the row program and the whole fit.
//!
//! Every arm is run in BOTH slope frames: the four-primary static frame
//! `(q₀, q₁, q̇₁, g)` and the six-primary follow-up-varying frame
//! `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` that #2765 introduced. A ψ calculus that is right
//! in one frame says nothing about the other — the pullback out of primary
//! space is a different map in each.
//!
//! The perturbation is exact by construction in both lanes (a design axis is
//! exactly linear in ψ; a baseline axis re-derives the chart at the displaced
//! coordinate), so the only error in the oracle is its own truncation, which a
//! Richardson pair certifies in-test rather than assuming.

use super::*;
use crate::custom_family::{
    BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyHyperLayout,
};
use crate::survival::construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, build_survival_marginal_slope_baseline_geometry,
    survival_baseline_config_from_theta,
};
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

const N_ROWS: usize = 24;

/// A scalar unit latent-score covariance, matching the shipped fixtures.
fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
            .expect("a 1x1 unit latent-score covariance"),
    )
}

/// Which outer ψ coordinate is under test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PsiAxis {
    /// A design-moving ψ on block 1 — the marginal (population index) design.
    MarginalDesign,
    /// A design-moving ψ on block 2 — the slope design.
    SlopeDesign,
    /// A baseline-chart coordinate. Moves the offset channels of the location
    /// index on every row and touches no design.
    Baseline(usize),
}

/// Which slope frame the row program runs in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlopeFrame {
    /// Four primaries `(q₀, q₁, q̇₁, g)` — a time-constant slope.
    Static,
    /// Six primaries `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` — a follow-up-varying slope.
    FollowUpVarying,
    /// Four primaries on a DECLARED skewed latent law (gam#2923): the location
    /// channels are anchored by the marginal identity instead of lowered in
    /// closed form, so the frame is nonlinear in every primary and every
    /// pullback below reads the implicit-function derivatives.
    Anchored,
}

impl SlopeFrame {
    fn label(self) -> &'static str {
        match self {
            Self::Static => "static-slope",
            Self::FollowUpVarying => "follow-up-varying-slope",
            Self::Anchored => "anchored-slope",
        }
    }
}

/// The declared law of the anchored frame: a two-component skewed law on 41
/// nodes, nothing a Gaussian describes.
fn anchored_law(n: usize) -> Arc<SurvivalLatentLaw> {
    let nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&u| {
            (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                + 0.35 * (-0.5 * ((u - 1.4) / 0.9).powi(2)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    let grid = crate::bms::EmpiricalZGrid::new(
        nodes,
        raw.into_iter().map(|w| w / total).collect(),
        "psi fd anchored law",
    )
    .expect("a valid declared law");
    Arc::new(
        SurvivalLatentLaw::from_kind(&crate::bms::LatentMeasureKind::GlobalEmpirical { grid }, n)
            .expect("materialise the declared law")
            .expect("an empirical law is a law"),
    )
}

// ── The fixture, as explicit functions of the displacement ──────────────────

fn age_entry() -> Array1<f64> {
    Array1::from_shape_fn(N_ROWS, |row| 0.25 + 0.10 * row as f64)
}

fn age_exit() -> Array1<f64> {
    let entry = age_entry();
    Array1::from_shape_fn(N_ROWS, |row| entry[row] + 0.75 + 0.03 * row as f64)
}

/// The baseline chart this fixture differentiates. Gompertz-Makeham carries
/// three coordinates whose partials are already FD-certified one layer down
/// (`marginal_slope_baseline_theta_partials_match_fd_for_gompertz_makeham`), so
/// a failure here is in the pullback, not in the chart.
fn baseline_config() -> SurvivalBaselineConfig {
    SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    }
}

fn base_marginal_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.30 + 0.40 * t,
            _ => -0.20 + 0.55 * (2.3 * t).sin(),
        }
    })
}

fn base_slope_exit_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.20 + 0.50 * t,
            _ => 0.15 * (1.7 * t + 0.4).cos(),
        }
    })
}

/// The entry-time margin of a follow-up-varying slope. Deliberately unequal to
/// the exit design: `g₀ = g₁` is exactly the degeneracy the static frame
/// already covers.
fn base_slope_entry_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.14 + 0.31 * t,
            _ => 0.09 * (2.1 * t + 0.7).cos(),
        }
    })
}

/// The exit-RATE margin `∂B(log t)/∂t` of a follow-up-varying slope.
fn base_slope_rate_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.07 - 0.05 * t,
            _ => 0.04 + 0.06 * (1.3 * t).sin(),
        }
    })
}

fn marginal_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.40 + 0.30 * (3.1 * t).cos(),
            _ => -0.25 + 0.45 * t,
        }
    })
}

fn slope_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.35 - 0.20 * t,
            _ => 0.10 + 0.30 * (2.7 * t).sin(),
        }
    })
}

/// The design derivative of the follow-up ENTRY margin.
///
/// A design ψ is one length scale shared by the whole covariate factor, so when
/// the slope carries a time margin the SAME ψ moves all three of its channels.
/// Handing the derivative only to the exit channel would differentiate a model
/// nobody fitted.
fn slope_entry_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.24 - 0.13 * t,
            _ => 0.06 + 0.21 * (2.7 * t).sin(),
        }
    })
}

fn slope_rate_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.12 - 0.07 * t,
            _ => 0.02 + 0.10 * (2.7 * t).sin(),
        }
    })
}

/// The family at ψ displacement `t` along `axis`, in `frame`.
///
/// Every input the displaced coordinate does NOT own is held fixed, so `t`
/// moves exactly what the ψ coordinate owns and nothing else.
fn family_at(axis: PsiAxis, frame: SlopeFrame, t: f64) -> SurvivalMarginalSlopeFamily {
    let n = N_ROWS;
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * ((i * 17 + 5) % n) as f64 / (n as f64)));

    // Baseline chart: displaced along its own coordinate when that is the axis
    // under test, and re-derived from the displaced coordinate rather than
    // perturbed in place — the chart's own encoder/decoder round trip is part
    // of what the outer solve walks.
    let mut config = baseline_config();
    if let PsiAxis::Baseline(index) = axis {
        let mut theta = crate::survival::construction::survival_baseline_theta_from_config(&config)
            .expect("baseline theta")
            .expect("Gompertz-Makeham carries a baseline chart");
        theta[index] += t;
        config = survival_baseline_config_from_theta(config.target, &theta)
            .expect("the displaced baseline coordinate stays in the chart's domain");
    }
    let geometry = Arc::new(
        build_survival_marginal_slope_baseline_geometry(&age_entry(), &age_exit(), &config)
            .expect("build baseline geometry")
            .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );

    let mut marginal = base_marginal_design();
    let mut slope_exit = base_slope_exit_design();
    let mut slope_entry = base_slope_entry_design();
    let mut slope_rate = base_slope_rate_design();
    match axis {
        PsiAxis::MarginalDesign => marginal.scaled_add(t, &marginal_design_derivative()),
        PsiAxis::SlopeDesign => {
            slope_exit.scaled_add(t, &slope_design_derivative());
            slope_entry.scaled_add(t, &slope_entry_design_derivative());
            slope_rate.scaled_add(t, &slope_rate_design_derivative());
        }
        PsiAxis::Baseline(_) => {}
    }

    let slope_layout: SlopeLayout = match frame {
        SlopeFrame::Static | SlopeFrame::Anchored => (DesignMatrix::from(slope_exit)).into(),
        SlopeFrame::FollowUpVarying => {
            let layout: SlopeLayout = (DesignMatrix::from(slope_exit)).into();
            layout
                .with_follow_up(
                    DesignMatrix::from(slope_entry),
                    DesignMatrix::from(slope_rate),
                )
                .expect("a shared slope layout accepts a follow-up margin")
        }
    };

    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: match frame {
            SlopeFrame::Anchored => Some(anchored_law(n)),
            SlopeFrame::Static | SlopeFrame::FollowUpVarying => None,
        },
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
            .expect("install the baseline chart as the family's hyper state"),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(geometry.offset_entry.clone()),
        offset_exit: Arc::new(geometry.offset_exit.clone()),
        derivative_offset_exit: Arc::new(geometry.derivative_offset_exit.clone()),
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        marginal_design: DesignMatrix::from(marginal),
        slope_layout,
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
    }
}

fn marginal_beta() -> Array1<f64> {
    ndarray::array![0.35, -0.18]
}

fn slope_beta() -> Array1<f64> {
    ndarray::array![0.22, 0.13]
}

/// Block states at fixed β for a given family.
///
/// `η` is a cached function of `(X, β)`, so it MUST be rebuilt at each
/// displacement: holding β fixed is the contract, holding a stale `η` fixed
/// would difference a different function from the one the ψ terms
/// differentiate.
fn states_at(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockState> {
    let m_beta = marginal_beta();
    let g_beta = slope_beta();
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(family.n),
        },
        ParameterBlockState {
            eta: m_design.dot(&m_beta),
            beta: m_beta,
        },
        ParameterBlockState {
            eta: g_design.dot(&g_beta),
            beta: g_beta,
        },
    ]
}

fn specs_for(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockSpec> {
    vec![
        fd_blockspec(0),
        fd_blockspec(family.marginal_design.ncols()),
        fd_blockspec(family.slope_layout.coefficient_design().ncols()),
    ]
}

fn fd_blockspec(cols: usize) -> ParameterBlockSpec {
    use std::sync::atomic::Ordering;
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let idx = SEQ.fetch_add(1, Ordering::Relaxed);
    ParameterBlockSpec {
        name: format!("psi_fd_{idx}"),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(cols)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// One design-moving ψ axis, carrying the analytic `X_ψ` that `family_at`
/// realizes exactly.
fn derivative_blocks(axis: PsiAxis) -> Vec<Vec<CustomFamilyBlockPsiDerivative>> {
    let x_psi = match axis {
        PsiAxis::MarginalDesign => marginal_design_derivative(),
        PsiAxis::SlopeDesign => slope_design_derivative(),
        PsiAxis::Baseline(_) => panic!("a baseline ψ axis owns no design derivative"),
    };
    let entry = vec![CustomFamilyBlockPsiDerivative::new(
        None,
        x_psi,
        Array2::zeros((2, 2)),
        None,
        None,
        None,
        None,
    )];
    match axis {
        PsiAxis::MarginalDesign => vec![Vec::new(), entry, Vec::new()],
        PsiAxis::SlopeDesign => vec![Vec::new(), Vec::new(), entry],
        PsiAxis::Baseline(_) => unreachable!(),
    }
}

fn hyper_layout(axis: PsiAxis) -> CustomFamilyHyperLayout {
    CustomFamilyHyperLayout::new(derivative_blocks(axis), Vec::new(), Array1::zeros(1))
        .expect("one design ψ axis")
}

/// The family's own `(objective, score, Hessian)` triple at displacement `t`,
/// in the NEGATIVE-log-likelihood sign the ψ terms are stated in.
///
/// All three come from the hooks the outer assembly reads, so a derivative that
/// matches these matches the function the criterion is actually built on. The
/// sign is not cosmetic and is worth stating once:
///
/// * `psi_terms_inner` accumulates the row program's `(nll, ∂nll/∂π, ∂²nll/∂π²)`
///   tower (`rigid_row_order2` returns the row NEGATIVE log-likelihood), and
///   `build_psi_hyper_coords` folds its `objective_psi` into `a` beside
///   `+½ βᵀS_ψβ` — so `a` is `∂_ψ(NLL + penalty)`, the criterion's own sign;
/// * `exact_newton_joint_gradient_evaluation` publishes the opposite one:
///   `log_likelihood = −Σ nll` and `gradient = −row_kernel_gradient = +∇_β ℓ`;
/// * `exact_newton_joint_hessian` publishes `H = −∇²_β ℓ`, i.e. already NLL-signed
///   (its trait doc states `H = -∇² log L`).
///
/// Differencing two of the three in one convention and the third in the other
/// would fail this gate by an exact factor of `−1` and say nothing about the
/// derivative, so the conversion happens here, once.
fn objective_score_hessian(
    axis: PsiAxis,
    frame: SlopeFrame,
    t: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(axis, frame, t);
    let states = states_at(&family);
    let specs = specs_for(&family);
    let evaluation = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("joint gradient evaluation")
        .expect("survival marginal-slope publishes a joint gradient evaluation");
    let hessian = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("survival marginal-slope publishes an explicit joint hessian");
    (-evaluation.log_likelihood, -evaluation.gradient, hessian)
}

/// Central difference plus its Richardson partner, so the gate reports the
/// oracle's own truncation instead of assuming it.
struct Ridders {
    value: f64,
    uncertainty: f64,
}

fn ridders(coarse: f64, fine: f64) -> Ridders {
    // Central differences are `O(h²)`: the `h/2` estimate carries a quarter of
    // the coarse remainder, so `(4·fine − coarse)/3` cancels it and the
    // difference of the two raw estimates bounds what is left.
    Ridders {
        value: (4.0 * fine - coarse) / 3.0,
        uncertainty: (fine - coarse).abs() / 3.0,
    }
}

struct PsiFiniteDifference {
    objective: Ridders,
    score: Vec<Ridders>,
    hessian: Vec<Ridders>,
    dim: usize,
}

fn finite_difference(axis: PsiAxis, frame: SlopeFrame, h: f64) -> PsiFiniteDifference {
    let coarse_plus = objective_score_hessian(axis, frame, h);
    let coarse_minus = objective_score_hessian(axis, frame, -h);
    let fine_plus = objective_score_hessian(axis, frame, 0.5 * h);
    let fine_minus = objective_score_hessian(axis, frame, -0.5 * h);

    let dim = coarse_plus.1.len();
    let objective = ridders(
        (coarse_plus.0 - coarse_minus.0) / (2.0 * h),
        (fine_plus.0 - fine_minus.0) / h,
    );
    let score = (0..dim)
        .map(|i| {
            ridders(
                (coarse_plus.1[i] - coarse_minus.1[i]) / (2.0 * h),
                (fine_plus.1[i] - fine_minus.1[i]) / h,
            )
        })
        .collect();
    let hessian = (0..dim * dim)
        .map(|flat| {
            let (r, c) = (flat / dim, flat % dim);
            ridders(
                (coarse_plus.2[[r, c]] - coarse_minus.2[[r, c]]) / (2.0 * h),
                (fine_plus.2[[r, c]] - fine_minus.2[[r, c]]) / h,
            )
        })
        .collect();
    PsiFiniteDifference {
        objective,
        score,
        hessian,
        dim,
    }
}

fn analytic_terms(axis: PsiAxis, frame: SlopeFrame) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(axis, frame, 0.0);
    let states = states_at(&family);
    let terms = match axis {
        PsiAxis::Baseline(index) => family
            .baseline_exact_joint_psi_terms_with_options(
                &states,
                index,
                &BlockwiseFitOptions::default(),
            )
            .expect("analytic baseline ψ terms")
            .expect("a nonlinear baseline chart publishes ψ terms"),
        design_axis => {
            let layout = hyper_layout(design_axis);
            family
                .psi_terms(&states, layout.design_derivative_blocks(), 0)
                .expect("analytic design ψ terms")
                .expect("a design ψ axis on a supported block publishes terms")
        }
    };
    let total: usize = states.iter().map(|state| state.beta.len()).sum();
    let hessian = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => terms.hessian_psi.clone(),
    };
    (terms.objective_psi, terms.score_psi.clone(), hessian)
}

/// Grade `analytic` against a Ridders-certified `fd`, refusing to charge a gap
/// to the analytic term unless the oracle's own uncertainty is well below it.
fn assert_matches(label: &str, analytic: f64, fd: &Ridders, scale: f64) {
    let gap = (analytic - fd.value).abs();
    let denominator = scale.max(analytic.abs()).max(fd.value.abs()).max(1e-12);
    assert!(
        fd.uncertainty <= 0.05 * denominator,
        "{label}: the finite-difference oracle did not resolve this component \
         (value={:.6e} uncertainty={:.3e} scale={denominator:.3e}); the analytic \
         term is not on trial here",
        fd.value,
        fd.uncertainty,
    );
    assert!(
        gap <= 1e-5 * denominator + 4.0 * fd.uncertainty,
        "{label}: analytic={analytic:.9e} fd={:.9e} gap={gap:.3e} \
         rel={:.3e} oracle_uncertainty={:.3e}",
        fd.value,
        gap / denominator,
        fd.uncertainty,
    );
}

fn max_abs<'a>(values: impl Iterator<Item = &'a f64>) -> f64 {
    values.fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn run_first_order_gate(axis: PsiAxis, frame: SlopeFrame) {
    let (objective_psi, score_psi, hessian_psi) = analytic_terms(axis, frame);
    let fd = finite_difference(axis, frame, 1e-3);
    let tag = format!("{axis:?}/{}", frame.label());
    assert_eq!(
        score_psi.len(),
        fd.dim,
        "{tag}: the ψ score must live in the flattened joint coefficient space"
    );
    assert_eq!(hessian_psi.dim(), (fd.dim, fd.dim), "{tag}: ψ Hessian shape");

    assert_matches(
        &format!("{tag} objective_psi"),
        objective_psi,
        &fd.objective,
        objective_psi.abs().max(fd.objective.value.abs()),
    );

    let score_scale = max_abs(score_psi.iter()).max(
        fd.score
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (i, oracle) in fd.score.iter().enumerate() {
        assert_matches(
            &format!("{tag} score_psi[{i}]"),
            score_psi[i],
            oracle,
            score_scale,
        );
    }

    let hessian_scale = max_abs(hessian_psi.iter()).max(
        fd.hessian
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (flat, oracle) in fd.hessian.iter().enumerate() {
        let (r, c) = (flat / fd.dim, flat % fd.dim);
        assert_matches(
            &format!("{tag} hessian_psi[{r},{c}]"),
            hessian_psi[[r, c]],
            oracle,
            hessian_scale,
        );
    }
}

// ── Design ψ: the matern/duchon length-scale lane ───────────────────────────

/// The marginal (population-index) design ψ axis in the four-primary frame.
#[test]
fn marginal_design_psi_terms_match_finite_difference_static_2765() {
    run_first_order_gate(PsiAxis::MarginalDesign, SlopeFrame::Static);
}

/// The slope design ψ axis in the four-primary frame. Filed separately from
/// the marginal axis because the two enter the row program through different
/// primary channels (`q₀,q₁` versus `g`), so one being right says nothing about
/// the other.
#[test]
fn slope_design_psi_terms_match_finite_difference_static_2765() {
    run_first_order_gate(PsiAxis::SlopeDesign, SlopeFrame::Static);
}

/// The marginal design ψ axis with a follow-up-varying slope. The location
/// index is unchanged by the frame, but the pullback out of primary space is
/// not: `∂H/∂ψ` now has three slope channels to cross against.
#[test]
fn marginal_design_psi_terms_match_finite_difference_follow_up_2765() {
    run_first_order_gate(PsiAxis::MarginalDesign, SlopeFrame::FollowUpVarying);
}

/// A design ψ on the slope surface of a follow-up-varying layout that carries
/// NO stored margin must be REFUSED, by name, rather than lowered through one of
/// its three channels.
///
/// This fixture's channel designs are arbitrary matrices, not tensor products
/// of a covariate factor with a recorded margin, so there is nothing to lift the
/// covariate derivative from, and any answer would be the derivative of a model
/// nobody asked for. A layout built from a margin lifts the derivative instead;
/// see `slope_design_psi_on_a_follow_up_margin_matches_finite_difference_2767`.
#[test]
fn a_slope_design_psi_on_a_follow_up_varying_slope_is_refused_2765() {
    let family = family_at(PsiAxis::SlopeDesign, SlopeFrame::FollowUpVarying, 0.0);
    let states = states_at(&family);
    let layout = hyper_layout(PsiAxis::SlopeDesign);
    let error = family
        .psi_terms(&states, layout.design_derivative_blocks(), 0)
        .expect_err("a slope design ψ has no lowering into the follow-up frame");
    assert!(
        error.contains("follow-up-varying slope"),
        "the refusal must name the surface it refuses: {error}"
    );
}

// ── Baseline ψ: the chart-coordinate lane #2765's fit dies on ───────────────

/// The baseline chart's first coordinate in the four-primary frame.
///
/// This is the lane #2765's acceptance fit refuses in: its search direction is
/// essentially pure baseline-ψ, and fifty successively halved steps all fail
/// Armijo. A gradient that disagrees with the criterion reproduces exactly that
/// signature, and nothing between the chart partials (already FD-certified) and
/// the whole fit had ever measured it.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis0_2765() {
    run_first_order_gate(PsiAxis::Baseline(0), SlopeFrame::Static);
}

/// The baseline chart's second coordinate — the Gompertz shape, which enters
/// the hazard through an exponential rather than a scale factor.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis1_2765() {
    run_first_order_gate(PsiAxis::Baseline(1), SlopeFrame::Static);
}

/// The baseline chart's third coordinate — the Makeham constant.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis2_2765() {
    run_first_order_gate(PsiAxis::Baseline(2), SlopeFrame::Static);
}

/// The baseline chart in the six-primary frame.
///
/// The baseline moves only the location index's offsets, so its primary
/// direction is zero on every slope channel in either frame — but the objects
/// it produces (`∂_ψ∇_βℓ̄` and `∂_ψ∇²_βℓ̄`) are NOT, and they have to be pulled
/// back through all three slope channels rather than one.
#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis0_2765() {
    run_first_order_gate(PsiAxis::Baseline(0), SlopeFrame::FollowUpVarying);
}

#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis1_2765() {
    run_first_order_gate(PsiAxis::Baseline(1), SlopeFrame::FollowUpVarying);
}

#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis2_2765() {
    run_first_order_gate(PsiAxis::Baseline(2), SlopeFrame::FollowUpVarying);
}

// ── The anchored frame (gam#2923): every ψ lane on a declared skewed law ─────
//
// The anchored frame's feature map is nonlinear in every primary — `α(q, b)`
// in the location channels as well as the slope — so each pullback below reads
// the implicit-function derivatives of the anchoring equation rather than a
// constant curvature. A ψ calculus that is right on the Gaussian frame says
// nothing about this one.

#[test]
fn marginal_design_psi_terms_match_finite_difference_anchored_2923() {
    run_first_order_gate(PsiAxis::MarginalDesign, SlopeFrame::Anchored);
}

#[test]
fn slope_design_psi_terms_match_finite_difference_anchored_2923() {
    run_first_order_gate(PsiAxis::SlopeDesign, SlopeFrame::Anchored);
}

#[test]
fn baseline_psi_terms_match_finite_difference_anchored_axis0_2923() {
    run_first_order_gate(PsiAxis::Baseline(0), SlopeFrame::Anchored);
}

#[test]
fn baseline_psi_terms_match_finite_difference_anchored_axis1_2923() {
    run_first_order_gate(PsiAxis::Baseline(1), SlopeFrame::Anchored);
}

#[test]
fn baseline_psi_terms_match_finite_difference_anchored_axis2_2923() {
    run_first_order_gate(PsiAxis::Baseline(2), SlopeFrame::Anchored);
}

// ── The frame reduction: no finite difference required ──────────────────────

/// A follow-up-varying layout whose margin is CONSTANT in time **is** a
/// time-constant slope, and the two frames must publish the same ψ terms.
///
/// With `B_entry = B_exit` and `B′_exit = 0` the six-primary frame satisfies
/// `g₀ = g₁` and `ġ₁ = 0` identically, so the model is literally the static
/// one — no approximation, no limit. This is the reduction the #2765 claim
/// comment promised ("the static fit must come out bit-identical where the time
/// basis is constant"), and it is the sharpest available check on the ψ lane
/// because it needs no oracle: a site that reads the slope through one channel
/// index gets `g₀`'s design and drops `g₁`'s, and here those two designs are
/// EQUAL, so the answer is short by exactly the dropped channels' contribution
/// rather than by something that could be argued to be small.
#[test]
fn a_constant_follow_up_margin_reproduces_the_static_frame_2765() {
    let static_family = family_at(PsiAxis::Baseline(0), SlopeFrame::Static, 0.0);
    let mut degenerate = family_at(PsiAxis::Baseline(0), SlopeFrame::Static, 0.0);
    let exit = degenerate
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    degenerate.slope_layout = degenerate
        .slope_layout
        .clone()
        .with_follow_up(
            DesignMatrix::from(exit),
            DesignMatrix::from(Array2::zeros((N_ROWS, 2))),
        )
        .expect("a constant margin is a valid follow-up layout");
    assert!(
        degenerate.slope_layout.is_follow_up_varying(),
        "the degenerate layout must still take the six-primary frame, or this \
         test compares the static path with itself"
    );

    for axis in 0..3 {
        let options = BlockwiseFitOptions::default();
        let want = static_family
            .baseline_exact_joint_psi_terms_with_options(
                &states_at(&static_family),
                axis,
                &options,
            )
            .expect("static baseline ψ terms")
            .expect("terms");
        let got = degenerate
            .baseline_exact_joint_psi_terms_with_options(
                &states_at(&degenerate),
                axis,
                &options,
            )
            .expect("degenerate follow-up baseline ψ terms")
            .expect("terms");

        let total = states_at(&static_family)
            .iter()
            .map(|state| state.beta.len())
            .sum::<usize>();
        let dense = |terms: &ExactNewtonJointPsiTerms| match terms.hessian_psi_operator.as_ref() {
            Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
            None => terms.hessian_psi.clone(),
        };
        let (want_h, got_h) = (dense(&want), dense(&got));

        let scale = want.objective_psi.abs().max(1e-12);
        assert!(
            (want.objective_psi - got.objective_psi).abs() <= 1e-12 * scale,
            "baseline axis {axis}: objective_psi differs between frames on a \
             CONSTANT margin: static={:.9e} follow-up={:.9e}",
            want.objective_psi,
            got.objective_psi,
        );
        let score_scale = max_abs(want.score_psi.iter()).max(1e-12);
        for i in 0..want.score_psi.len() {
            assert!(
                (want.score_psi[i] - got.score_psi[i]).abs() <= 1e-10 * score_scale,
                "baseline axis {axis}: score_psi[{i}] differs between frames on a \
                 CONSTANT margin: static={:.9e} follow-up={:.9e}",
                want.score_psi[i],
                got.score_psi[i],
            );
        }
        let hessian_scale = max_abs(want_h.iter()).max(1e-12);
        for r in 0..total {
            for c in 0..total {
                assert!(
                    (want_h[[r, c]] - got_h[[r, c]]).abs() <= 1e-10 * hessian_scale,
                    "baseline axis {axis}: hessian_psi[{r},{c}] differs between frames \
                     on a CONSTANT margin: static={:.9e} follow-up={:.9e}",
                    want_h[[r, c]],
                    got_h[[r, c]],
                );
            }
        }
    }
}

// ── The degenerate end of the same contract ─────────────────────────────────

/// A ψ axis whose design derivative is exactly zero must publish exactly zero
/// terms — the one case where the finite difference is exact rather than
/// certified, and the cheapest possible check that the ψ lane is reading its
/// own input.
#[test]
fn a_zero_psi_design_derivative_publishes_zero_terms_2765() {
    let family = family_at(PsiAxis::MarginalDesign, SlopeFrame::Static, 0.0);
    let states = states_at(&family);
    let blocks = vec![
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((N_ROWS, 2)),
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let terms = family
        .psi_terms(&states, &blocks, 0)
        .expect("zero ψ terms")
        .expect("a design ψ axis publishes terms even when its derivative vanishes");
    assert_eq!(terms.objective_psi, 0.0);
    assert!(terms.score_psi.iter().all(|value| *value == 0.0));
    let total: usize = states.iter().map(|state| state.beta.len()).sum();
    let hessian = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => terms.hessian_psi.clone(),
    };
    assert!(hessian.iter().all(|value| *value == 0.0));
}

// ── A slope design ψ on a follow-up margin ───────────────────────────────────
//
// With a time margin the slope's channel designs are `X_cov ⊗ B_entry`,
// `X_cov ⊗ B_exit` and `X_cov ⊗ Ḃ_exit`, and a length-scale ψ moves `X_cov`. The
// family lifts the covariate derivative row onto all three channels from the
// layout's stored margin (gam#2767), so the fixture below builds each channel as a
// genuine tensor product and differences the family's own objective, score and
// Hessian as `X_cov` moves.

/// The follow-up-varying family whose slope covariate factor is displaced by `t`
/// along the slope design derivative, and its block states at fixed β.
fn tensored_slope_family_at(t: f64) -> (SurvivalMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let mut family = family_at(PsiAxis::Baseline(0), SlopeFrame::Static, 0.0);
    let mut covariate = base_slope_exit_design();
    covariate.scaled_add(t, &slope_design_derivative());
    let margin = SlopeTimeMargin {
        entry: Arc::new(base_slope_entry_design()),
        exit: Arc::new(base_marginal_design()),
        derivative_exit: Arc::new(base_slope_rate_design()),
    };
    let tensor = |basis: &Array2<f64>| {
        Array2::from_shape_fn((N_ROWS, 4), |(row, column)| {
            covariate[[row, column / 2]] * basis[[row, column % 2]]
        })
    };
    let layout: SlopeLayout = DesignMatrix::from(tensor(&margin.exit)).into();
    family.slope_layout = layout
        .with_follow_up(
            DesignMatrix::from(tensor(&margin.entry)),
            DesignMatrix::from(tensor(&margin.derivative_exit)),
        )
        .expect("a shared slope layout accepts a follow-up margin")
        .with_follow_up_time_margin(margin.clone())
        .expect("the tensored layout carries its margin");
    let m_beta = marginal_beta();
    let g_beta = ndarray::array![0.22, 0.13, -0.09, 0.17];
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(N_ROWS),
        },
        ParameterBlockState {
            eta: family.marginal_design.to_dense().to_owned().dot(&m_beta),
            beta: m_beta,
        },
        ParameterBlockState {
            eta: family
                .slope_layout
                .coefficient_design()
                .to_dense()
                .to_owned()
                .dot(&g_beta),
            beta: g_beta,
        },
    ];
    (family, states)
}

#[test]
fn slope_design_psi_on_a_follow_up_margin_matches_finite_difference_2767() {
    let blocks = vec![
        Vec::new(),
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative::new(
            None,
            slope_design_derivative(),
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
    ];
    let (family, states) = tensored_slope_family_at(0.0);
    assert!(
        family.slope_layout.is_follow_up_varying(),
        "the fixture must run the six-primary frame"
    );
    let terms = family
        .psi_terms(&states, &blocks, 0)
        .expect("follow-up slope design ψ terms")
        .expect("a slope design ψ on a follow-up margin publishes terms");
    let total: usize = states.iter().map(|state| state.beta.len()).sum();
    let hessian_psi = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => terms.hessian_psi.clone(),
    };

    let evaluate = |t: f64| {
        let (family, states) = tensored_slope_family_at(t);
        let specs = specs_for(&family);
        let evaluation = family
            .exact_newton_joint_gradient_evaluation(&states, &specs)
            .expect("joint gradient evaluation")
            .expect("survival marginal-slope publishes a joint gradient evaluation");
        let hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("survival marginal-slope publishes an explicit joint hessian");
        (-evaluation.log_likelihood, -evaluation.gradient, hessian)
    };
    let h = 1e-3;
    let (coarse_plus, coarse_minus) = (evaluate(h), evaluate(-h));
    let (fine_plus, fine_minus) = (evaluate(0.5 * h), evaluate(-0.5 * h));
    let oracle = |plus: f64, minus: f64, fine_p: f64, fine_m: f64| {
        ridders((plus - minus) / (2.0 * h), (fine_p - fine_m) / h)
    };

    assert_matches(
        "follow-up slope design objective_psi",
        terms.objective_psi,
        &oracle(coarse_plus.0, coarse_minus.0, fine_plus.0, fine_minus.0),
        terms.objective_psi.abs(),
    );
    let score_scale = max_abs(terms.score_psi.iter()).max(1e-12);
    for i in 0..total {
        assert_matches(
            &format!("follow-up slope design score_psi[{i}]"),
            terms.score_psi[i],
            &oracle(coarse_plus.1[i], coarse_minus.1[i], fine_plus.1[i], fine_minus.1[i]),
            score_scale,
        );
    }
    let hessian_scale = max_abs(hessian_psi.iter()).max(1e-12);
    for r in 0..total {
        for c in 0..total {
            assert_matches(
                &format!("follow-up slope design hessian_psi[{r},{c}]"),
                hessian_psi[[r, c]],
                &oracle(
                    coarse_plus.2[[r, c]],
                    coarse_minus.2[[r, c]],
                    fine_plus.2[[r, c]],
                    fine_minus.2[[r, c]],
                ),
                hessian_scale,
            );
        }
    }
}

// ── The learned frailty scale: log σ ─────────────────────────────────────────
//
// A learned Gaussian-shift frailty scale is an outer coordinate `log σ`. Its ψ
// terms are the family's own objective, score and joint Hessian differentiated in
// `log σ` at fixed β, and in the follow-up frame the scale moves all three slope
// channels (gam#2767). The time-constant frame runs the compiled lowering and the
// follow-up frame the parameter-jet one; the gates difference both against the
// family's own hooks.

/// The drift fixture carrying a Gaussian-shift frailty scale `σ = exp(log_sigma)`.
fn frailty_family_at(
    frame: SlopeFrame,
    log_sigma: f64,
) -> (SurvivalMarginalSlopeFamily, Array1<f64>) {
    let (mut family, beta) = drift_family_and_states(frame);
    family.gaussian_frailty_sd = Some(log_sigma.exp());
    (family, beta)
}

fn drift_specs(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockSpec> {
    vec![
        fd_blockspec(family.design_exit.ncols()),
        fd_blockspec(family.marginal_design.ncols()),
        fd_blockspec(family.slope_layout.coefficient_design().ncols()),
    ]
}

/// The family's `(NLL, NLL score, joint Hessian)` at fixed β as `log σ` moves.
fn frailty_objective_score_hessian(
    frame: SlopeFrame,
    log_sigma: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    let (family, beta) = frailty_family_at(frame, log_sigma);
    let states = states_at_beta(&family, &beta);
    let evaluation = family
        .exact_newton_joint_gradient_evaluation(&states, &drift_specs(&family))
        .expect("joint gradient evaluation")
        .expect("survival marginal-slope publishes a joint gradient evaluation");
    let hessian = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("survival marginal-slope publishes an explicit joint hessian");
    (-evaluation.log_likelihood, -evaluation.gradient, hessian)
}

fn grade_log_sigma_terms(
    label: &str,
    objective: f64,
    score: &Array1<f64>,
    hessian: &Array2<f64>,
    lower_order_at: impl Fn(f64) -> (f64, Array1<f64>, Array2<f64>),
) {
    let h = 1e-3;
    let (coarse_plus, coarse_minus) = (lower_order_at(h), lower_order_at(-h));
    let (fine_plus, fine_minus) = (lower_order_at(0.5 * h), lower_order_at(-0.5 * h));
    let oracle = |plus: f64, minus: f64, fine_p: f64, fine_m: f64| {
        ridders((plus - minus) / (2.0 * h), (fine_p - fine_m) / h)
    };
    assert_matches(
        &format!("{label} objective"),
        objective,
        &oracle(coarse_plus.0, coarse_minus.0, fine_plus.0, fine_minus.0),
        objective.abs(),
    );
    let dim = score.len();
    let score_scale = max_abs(score.iter()).max(1e-12);
    for i in 0..dim {
        assert_matches(
            &format!("{label} score[{i}]"),
            score[i],
            &oracle(coarse_plus.1[i], coarse_minus.1[i], fine_plus.1[i], fine_minus.1[i]),
            score_scale,
        );
    }
    let hessian_scale = max_abs(hessian.iter()).max(1e-12);
    for r in 0..dim {
        for c in 0..dim {
            assert_matches(
                &format!("{label} hessian[{r},{c}]"),
                hessian[[r, c]],
                &oracle(
                    coarse_plus.2[[r, c]],
                    coarse_minus.2[[r, c]],
                    fine_plus.2[[r, c]],
                    fine_minus.2[[r, c]],
                ),
                hessian_scale,
            );
        }
    }
}

fn dense_psi_hessian(
    operator: Option<&Arc<dyn HyperOperator>>,
    dense: &Array2<f64>,
    total: usize,
) -> Array2<f64> {
    match operator {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => dense.clone(),
    }
}

/// `∂_{log σ}` of the family objective, score and joint Hessian, in both slope frames.
#[test]
fn log_sigma_psi_terms_match_finite_difference_2767() {
    let options = BlockwiseFitOptions::default();
    let log_sigma = 0.6_f64.ln();
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        let (family, beta) = frailty_family_at(frame, log_sigma);
        let states = states_at_beta(&family, &beta);
        let terms = family
            .sigma_exact_joint_psi_terms_with_options(&states, &drift_specs(&family), &options)
            .expect("log-sigma ψ terms")
            .expect("a rigid marginal-slope family publishes log-sigma ψ terms");
        let hessian = dense_psi_hessian(
            terms.hessian_psi_operator.as_ref(),
            &terms.hessian_psi,
            beta.len(),
        );
        grade_log_sigma_terms(
            &format!("{} log-sigma first order", frame.label()),
            terms.objective_psi,
            &terms.score_psi,
            &hessian,
            |t| frailty_objective_score_hessian(frame, log_sigma + t),
        );
    }
}

/// `∂²_{log σ}` of the family objective, score and joint Hessian, differenced from the
/// first-order terms, in both slope frames.
#[test]
fn log_sigma_psi_second_order_terms_match_finite_difference_2767() {
    let options = BlockwiseFitOptions::default();
    let log_sigma = 0.6_f64.ln();
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        let (family, beta) = frailty_family_at(frame, log_sigma);
        let states = states_at_beta(&family, &beta);
        let terms = family
            .sigma_exact_joint_psisecond_order_terms_with_options(&states, &options)
            .expect("log-sigma second-order ψ terms")
            .expect("a rigid marginal-slope family publishes log-sigma second-order terms");
        let total = beta.len();
        let hessian = dense_psi_hessian(
            terms.hessian_psi_psi_operator.as_ref(),
            &terms.hessian_psi_psi,
            total,
        );
        grade_log_sigma_terms(
            &format!("{} log-sigma second order", frame.label()),
            terms.objective_psi_psi,
            &terms.score_psi_psi,
            &hessian,
            |t| {
                let (family, beta) = frailty_family_at(frame, log_sigma + t);
                let states = states_at_beta(&family, &beta);
                let first = family
                    .sigma_exact_joint_psi_terms_with_options(
                        &states,
                        &drift_specs(&family),
                        &options,
                    )
                    .expect("log-sigma ψ terms")
                    .expect("a rigid marginal-slope family publishes log-sigma ψ terms");
                let hessian = dense_psi_hessian(
                    first.hessian_psi_operator.as_ref(),
                    &first.hessian_psi,
                    total,
                );
                (first.objective_psi, first.score_psi, hessian)
            },
        );
    }
}

// ── The β-drift of the joint Hessian: `D_β H[δ]` ────────────────────────────
//
// The outer criterion's `½ log|H(β̂(θ), θ) + S_λ|` has TWO derivative halves at
// every θ coordinate — the frozen one, `½ tr(K · ∂_θ H|_β)`, and the
// mode-response one, `½ tr(K · D_β H[dβ̂/dθ])`. The second is the ONLY atom in
// the whole criterion that reads `dβ̂/dθ`; `fixed_beta` is an envelope
// derivative at fixed β̂ and `logdet_s` never sees β at all. So when the
// `logdet_h` atom is the only one that disagrees with a finite difference of
// the criterion — which is exactly what #2765 measured, on EVERY ρ and ψ
// coordinate of the acceptance fixture, with `fixed_beta` right to six digits
// and `logdet_s` right to seven — `D_β H` is one of the two objects that can
// own it.
//
// Nothing differenced it. The shipped coverage for this hook is finiteness,
// subsample-vs-unsampled equality and batched-vs-per-axis agreement, every one
// of which a consistently wrong third derivative passes. These gates difference
// the family's OWN joint Hessian along the direction, which is the definition
// the outer trace contracts.

/// Block states at an arbitrary flat β, with every `η` rebuilt from it.
///
/// `η` is a cached function of `(X, β)`, so a β displacement that left `η`
/// stale would difference a different function from the one `D_β H` claims to
/// differentiate — the same contract [`states_at`] states for the ψ lane.
fn states_at_beta(
    family: &SurvivalMarginalSlopeFamily,
    beta_flat: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let t_cols = family.design_exit.ncols();
    let m_cols = family.marginal_design.ncols();
    let g_cols = family.slope_layout.coefficient_design().ncols();
    assert_eq!(
        beta_flat.len(),
        t_cols + m_cols + g_cols,
        "the fixture's flat β is the time block, then the marginal block, then the slope block"
    );
    let t_beta = beta_flat.slice(ndarray::s![..t_cols]).to_owned();
    let m_beta = beta_flat.slice(ndarray::s![t_cols..t_cols + m_cols]).to_owned();
    let g_beta = beta_flat.slice(ndarray::s![t_cols + m_cols..]).to_owned();
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    // The time block's own `eta` is the EXIT channel, matching the block spec
    // the fit installs; the entry and derivative channels are read from their
    // designs by the row program rather than from a cached `eta`.
    let t_eta = if t_cols == 0 {
        Array1::zeros(family.n)
    } else {
        family.design_exit.to_dense().to_owned().dot(&t_beta)
    };
    vec![
        ParameterBlockState {
            eta: t_eta,
            beta: t_beta,
        },
        ParameterBlockState {
            eta: m_design.dot(&m_beta),
            beta: m_beta,
        },
        ParameterBlockState {
            eta: g_design.dot(&g_beta),
            beta: g_beta,
        },
    ]
}

/// The joint Hessian at `β + t·δ`, in the same NLL sign the hook publishes.
fn joint_hessian_at(
    family: &SurvivalMarginalSlopeFamily,
    beta: &Array1<f64>,
    direction: &Array1<f64>,
    t: f64,
) -> Array2<f64> {
    let displaced = beta + &(direction * t);
    let states = states_at_beta(family, &displaced);
    family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("survival marginal-slope publishes an explicit joint hessian")
}

/// The drift fixture's TIME block: `q` gains a real `(1, log t)` surface on top
/// of the baseline chart's offsets.
///
/// [`family_at`] leaves the three time designs empty, which is the right fixture
/// for a ψ lane that never touches them — but it means the time↔marginal and
/// time↔slope halves of the pullback are never exercised. `D_β H` reads
/// every block, so its gate builds them: `X_entry = (1, log t₀)`,
/// `X_exit = (1, log t₁)` and `X_derivative = (0, 1/t₁)`, which is the exact
/// derivative pair the row program's monotonicity contract expects (`q̇₁ > 0`
/// for any positive slope coefficient).
fn time_designs() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let entry = age_entry();
    let exit = age_exit();
    let x_entry = Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        if col == 0 { 1.0 } else { entry[row].ln() }
    });
    let x_exit = Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        if col == 0 { 1.0 } else { exit[row].ln() }
    });
    let x_derivative = Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        if col == 0 { 0.0 } else { 1.0 / exit[row] }
    });
    (x_entry, x_exit, x_derivative)
}

fn time_beta() -> Array1<f64> {
    ndarray::array![0.12, 0.45]
}

/// [`family_at`] with a non-empty time block, and the block states that go with
/// it. The drift gate uses this rather than the ψ fixture so all six coefficient
/// blocks of the pullback — `tt`, `mm`, `gg`, `tm`, `tg`, `mg` — carry weight.
fn drift_family_and_states(
    frame: SlopeFrame,
) -> (SurvivalMarginalSlopeFamily, Array1<f64>) {
    let mut family = family_at(PsiAxis::MarginalDesign, frame, 0.0);
    let (x_entry, x_exit, x_derivative) = time_designs();
    family.design_entry = DesignMatrix::from(x_entry);
    family.design_exit = DesignMatrix::from(x_exit);
    family.design_derivative_exit = DesignMatrix::from(x_derivative);
    let mut beta = Array1::<f64>::zeros(6);
    beta.slice_mut(ndarray::s![..2]).assign(&time_beta());
    beta.slice_mut(ndarray::s![2..4]).assign(&marginal_beta());
    beta.slice_mut(ndarray::s![4..]).assign(&slope_beta());
    (family, beta)
}

fn run_beta_drift_gate(frame: SlopeFrame, direction: Array1<f64>, label: &str) {
    let (family, beta) = drift_family_and_states(frame);
    assert_eq!(
        direction.len(),
        beta.len(),
        "{label}: the drift direction lives in the same flat coefficient space as β"
    );
    let states = states_at_beta(&family, &beta);
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint Hessian directional derivative")
        .expect("survival marginal-slope publishes an exact D_beta H");
    let dim = beta.len();
    assert_eq!(analytic.dim(), (dim, dim), "{label}: D_beta H shape");

    let h = 1e-3;
    let coarse_plus = joint_hessian_at(&family, &beta, &direction, h);
    let coarse_minus = joint_hessian_at(&family, &beta, &direction, -h);
    let fine_plus = joint_hessian_at(&family, &beta, &direction, 0.5 * h);
    let fine_minus = joint_hessian_at(&family, &beta, &direction, -0.5 * h);

    let scale = max_abs(analytic.iter()).max(1e-12);
    for row in 0..dim {
        for column in 0..dim {
            let oracle = ridders(
                (coarse_plus[[row, column]] - coarse_minus[[row, column]]) / (2.0 * h),
                (fine_plus[[row, column]] - fine_minus[[row, column]]) / h,
            );
            assert_matches(
                &format!("{label}/{} D_beta H[{row},{column}]", frame.label()),
                analytic[[row, column]],
                &oracle,
                scale,
            );
        }
    }
}

/// `D_β H[δ]` along a direction that moves BOTH blocks, in the four-primary
/// static frame.
#[test]
fn beta_hessian_drift_matches_finite_difference_static_2765() {
    run_beta_drift_gate(
        SlopeFrame::Static,
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        "joint",
    );
}

/// The same, in the six-primary follow-up-varying frame. A drift that is right
/// in one frame says nothing about the other: the pullback out of primary space
/// is a different map in each, and #2765's whole subject is the second one.
#[test]
fn beta_hessian_drift_matches_finite_difference_follow_up_2765() {
    run_beta_drift_gate(
        SlopeFrame::FollowUpVarying,
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        "joint",
    );
}

/// A direction confined to the MARGINAL block, so a defect in the slope
/// pullback cannot mask (or be masked by) one in the marginal pullback.
#[test]
fn beta_hessian_drift_matches_finite_difference_marginal_direction_2765() {
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        run_beta_drift_gate(
            frame,
            ndarray::array![0.0, 0.0, 0.55, -0.31, 0.0, 0.0],
            "marginal-only",
        );
    }
}

/// A direction confined to the SLOPE block — the one #2765 generalized from
/// one channel to three.
#[test]
fn beta_hessian_drift_matches_finite_difference_slope_direction_2765() {
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        run_beta_drift_gate(
            frame,
            ndarray::array![0.0, 0.0, 0.0, 0.0, 0.47, 0.23],
            "slope-only",
        );
    }
}

/// A direction confined to the TIME block. The time↔slope half of the
/// pullback has its own channel loop, and a direction that never moves the
/// slope block is the only one that isolates it.
#[test]
fn beta_hessian_drift_matches_finite_difference_time_direction_2765() {
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        run_beta_drift_gate(
            frame,
            ndarray::array![0.29, -0.21, 0.0, 0.0, 0.0, 0.0],
            "time-only",
        );
    }
}

/// The oracle-free companion to the four `D_β H` gates: on a CONSTANT time
/// margin the follow-up frame IS the static one, so its drift must agree
/// entry-for-entry.
///
/// `B_entry = B_exit` and `B′_exit = 0` make `g₀ = g₁` and `ġ₁ = 0` identically,
/// which is the same reduction `a_constant_follow_up_margin_reproduces_the_static_frame_2765`
/// applies to the ψ terms. It is the sharpest check available on a pullback that
/// reads the slope through one channel index: with the two channel designs
/// EQUAL, a site that keeps only `g₀`'s design is short by exactly the dropped
/// channels' contribution rather than by something that could be argued small.
/// Before the pullback was converted this failed by `1.7e0` relative on the
/// marginal↔slope cross block.
#[test]
fn a_constant_follow_up_margin_reproduces_the_static_beta_drift_2765() {
    let (static_family, beta) = drift_family_and_states(SlopeFrame::Static);
    let mut degenerate = static_family.clone();
    let exit = degenerate
        .slope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    let width = exit.ncols();
    degenerate.slope_layout = degenerate
        .slope_layout
        .clone()
        .with_follow_up(
            DesignMatrix::from(exit),
            DesignMatrix::from(Array2::zeros((N_ROWS, width))),
        )
        .expect("a constant margin is a valid follow-up layout");
    assert!(
        degenerate.slope_layout.is_follow_up_varying(),
        "the degenerate layout must still take the six-primary frame, or this \
         test compares the static path with itself"
    );

    for direction in [
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        ndarray::array![0.0, 0.0, 0.0, 0.0, 0.47, 0.23],
        ndarray::array![0.29, -0.21, 0.0, 0.0, 0.0, 0.0],
    ] {
        let want = static_family
            .exact_newton_joint_hessian_directional_derivative(
                &states_at_beta(&static_family, &beta),
                &direction,
            )
            .expect("static D_beta H")
            .expect("terms");
        let got = degenerate
            .exact_newton_joint_hessian_directional_derivative(
                &states_at_beta(&degenerate, &beta),
                &direction,
            )
            .expect("degenerate follow-up D_beta H")
            .expect("terms");
        let scale = max_abs(want.iter()).max(1e-12);
        for row in 0..beta.len() {
            for column in 0..beta.len() {
                assert!(
                    (want[[row, column]] - got[[row, column]]).abs() <= 1e-10 * scale,
                    "D_beta H[{row},{column}] differs between frames on a CONSTANT \
                     margin along {direction:?}: static={:.9e} follow-up={:.9e}",
                    want[[row, column]],
                    got[[row, column]],
                );
            }
        }
    }
}

// ── `D²_β H[u, v]`: the SECOND directional derivative ───────────────────────
//
// The Jeffreys curvature `H_Φ` is a divided-difference object built from `H` and
// its first directional derivatives, so ITS β-drift `D_β H_Φ[δ]` consumes the
// family's SECOND directional derivatives `H²dot[δ, e_a]`. That drift is added
// to the likelihood drift before the outer criterion contracts
// `½ tr(K · D_β H[dβ̂/dθ])`, so a wrong `D²_β H` corrupts the same
// mode-response atom `D_β H` does — and, like `D_β H` before the four gates
// above, nothing differenced it.
//
// `D²_β H[u, v]` is symmetric in `(u, v)` and equals `∂/∂t D_β H[v]` at
// `β + t·u`, which is what these difference.

fn joint_drift_at(
    family: &SurvivalMarginalSlopeFamily,
    beta: &Array1<f64>,
    shift: &Array1<f64>,
    t: f64,
    direction: &Array1<f64>,
) -> Array2<f64> {
    let displaced = beta + &(shift * t);
    let states = states_at_beta(family, &displaced);
    family
        .exact_newton_joint_hessian_directional_derivative(&states, direction)
        .expect("joint Hessian directional derivative")
        .expect("survival marginal-slope publishes an exact D_beta H")
}

fn run_beta_second_drift_gate(frame: SlopeFrame, u: Array1<f64>, v: Array1<f64>, label: &str) {
    let (family, beta) = drift_family_and_states(frame);
    let states = states_at_beta(&family, &beta);
    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &v)
        .expect("joint Hessian second directional derivative")
        .expect("survival marginal-slope publishes an exact D2_beta H");
    let dim = beta.len();
    assert_eq!(analytic.dim(), (dim, dim), "{label}: D2_beta H shape");

    let h = 1e-3;
    let coarse_plus = joint_drift_at(&family, &beta, &u, h, &v);
    let coarse_minus = joint_drift_at(&family, &beta, &u, -h, &v);
    let fine_plus = joint_drift_at(&family, &beta, &u, 0.5 * h, &v);
    let fine_minus = joint_drift_at(&family, &beta, &u, -0.5 * h, &v);

    let scale = max_abs(analytic.iter()).max(1e-12);
    for row in 0..dim {
        for column in 0..dim {
            let oracle = ridders(
                (coarse_plus[[row, column]] - coarse_minus[[row, column]]) / (2.0 * h),
                (fine_plus[[row, column]] - fine_minus[[row, column]]) / h,
            );
            assert_matches(
                &format!("{label}/{} D2_beta H[{row},{column}]", frame.label()),
                analytic[[row, column]],
                &oracle,
                scale,
            );
        }
    }
}

#[test]
fn beta_hessian_second_drift_matches_finite_difference_static_2765() {
    run_beta_second_drift_gate(
        SlopeFrame::Static,
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        ndarray::array![-0.11, 0.31, 0.19, 0.24, -0.28, 0.14],
        "joint",
    );
}

#[test]
fn beta_hessian_second_drift_matches_finite_difference_follow_up_2765() {
    run_beta_second_drift_gate(
        SlopeFrame::FollowUpVarying,
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        ndarray::array![-0.11, 0.31, 0.19, 0.24, -0.28, 0.14],
        "joint",
    );
}

/// The anchored frame's joint second drift (gam#2923): the dense fourth-order
/// tower of the anchored feature map, contracted with two joint directions.
#[test]
fn beta_hessian_second_drift_matches_finite_difference_anchored_2923() {
    run_beta_second_drift_gate(
        SlopeFrame::Anchored,
        ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19],
        ndarray::array![-0.11, 0.31, 0.19, 0.24, -0.28, 0.14],
        "joint-anchored",
    );
}

/// A slope-only pair, so a defect in the slope channels' fourth-order
/// contraction cannot be masked by the time/marginal blocks.
#[test]
fn beta_hessian_second_drift_matches_finite_difference_slope_pair_2765() {
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying, SlopeFrame::Anchored] {
        run_beta_second_drift_gate(
            frame,
            ndarray::array![0.0, 0.0, 0.0, 0.0, 0.47, 0.23],
            ndarray::array![0.0, 0.0, 0.0, 0.0, -0.19, 0.35],
            "slope-pair",
        );
    }
}

// ── The mixed third information derivatives of a baseline coordinate ────────
//
// The explicit Jeffreys curvature differentiates `H_Φ` along ψ and β together, so
// for every baseline-chart coordinate θ it reads `D_β(D_β ∂_θ H[v])` and
// `D_β ∂²_θθ' H` along every coefficient axis. The survival ψ workspace had
// neither, and #2765's acceptance fit certified its outer search and then refused
// the `ValueGradientHessian` evaluation that came next ("exact third information
// derivatives are unavailable for psi axis 0"). Each object is differenced along
// every coefficient axis against the lower-order hook it differentiates; the gates
// above difference those hooks against the family's own Hessian.

fn grade_all_beta_axes(
    label: &str,
    analytic: &[Array2<f64>],
    beta: &Array1<f64>,
    lower_order_at: impl Fn(&Array1<f64>) -> Array2<f64>,
) {
    let dim = beta.len();
    assert_eq!(analytic.len(), dim, "{label}: one matrix per coefficient axis");
    let global_scale = analytic
        .iter()
        .fold(0.0_f64, |acc, matrix| acc.max(max_abs(matrix.iter())));
    let h = 1e-3;
    for (coefficient_axis, matrix) in analytic.iter().enumerate() {
        assert_eq!(matrix.dim(), (dim, dim), "{label}: axis {coefficient_axis} shape");
        let at = |t: f64| {
            let mut displaced = beta.clone();
            displaced[coefficient_axis] += t;
            lower_order_at(&displaced)
        };
        let (coarse_plus, coarse_minus) = (at(h), at(-h));
        let (fine_plus, fine_minus) = (at(0.5 * h), at(-0.5 * h));
        let scale = max_abs(matrix.iter())
            .max(1e-6 * global_scale)
            .max(1e-12);
        for row in 0..dim {
            for column in 0..dim {
                let oracle = ridders(
                    (coarse_plus[[row, column]] - coarse_minus[[row, column]]) / (2.0 * h),
                    (fine_plus[[row, column]] - fine_minus[[row, column]]) / h,
                );
                assert_matches(
                    &format!("{label} coefficient axis {coefficient_axis} [{row},{column}]"),
                    matrix[[row, column]],
                    &oracle,
                    scale,
                );
            }
        }
    }
}

/// `D_β(D_β ∂_θ H[v])` for every baseline coordinate, in both slope frames.
#[test]
fn baseline_psi_by_beta_third_information_matches_finite_difference_2765() {
    let options = BlockwiseFitOptions::default();
    let direction = ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19];
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying] {
        let (family, beta) = drift_family_and_states(frame);
        for baseline_axis in 0..3 {
            let analytic = family
                .baseline_exact_joint_psihessian_second_directional_derivative_all_beta_axes_with_options(
                    &states_at_beta(&family, &beta),
                    baseline_axis,
                    &direction,
                    &options,
                )
                .expect("baseline-by-coefficient third information derivative");
            grade_all_beta_axes(
                &format!("{}/baseline {baseline_axis} by beta", frame.label()),
                &analytic,
                &beta,
                |displaced| {
                    family
                        .baseline_exact_joint_psihessian_directional_derivative_with_options(
                            &states_at_beta(&family, displaced),
                            baseline_axis,
                            &direction,
                            &options,
                        )
                        .expect("baseline ψ Hessian drift")
                        .expect("a rigid baseline chart publishes its ψ Hessian drift")
                },
            );
        }
    }
}

/// `D_β(D_β ∂_ψ H[v])` for a design ψ: on the marginal surface in both slope frames,
/// and on the slope surface of a time-constant slope (the follow-up frame refuses it).
#[test]
fn design_psi_by_beta_third_information_matches_finite_difference_2765() {
    let options = BlockwiseFitOptions::default();
    let direction = ndarray::array![0.23, 0.17, 0.41, -0.27, 0.33, 0.19];
    for (axis, frame) in [
        (PsiAxis::MarginalDesign, SlopeFrame::Static),
        (PsiAxis::MarginalDesign, SlopeFrame::FollowUpVarying),
        (PsiAxis::SlopeDesign, SlopeFrame::Static),
    ] {
        let layout = hyper_layout(axis);
        let (family, beta) = drift_family_and_states(frame);
        let analytic = family
            .design_psi_hessian_second_directional_derivative_all_beta_axes_with_options(
                &states_at_beta(&family, &beta),
                layout.design_derivative_blocks(),
                0,
                &direction,
                &options,
            )
            .expect("design-by-coefficient third information derivative")
            .expect("a design ψ axis on a supported block publishes its third information derivative");
        grade_all_beta_axes(
            &format!("{}/{axis:?} by beta", frame.label()),
            &analytic,
            &beta,
            |displaced| {
                family
                    .psi_hessian_directional_derivative_with_options(
                        &states_at_beta(&family, displaced),
                        layout.design_derivative_blocks(),
                        0,
                        &direction,
                        &options,
                    )
                    .expect("design ψ Hessian drift")
                    .expect("a design ψ axis on a supported block publishes its Hessian drift")
            },
        );
    }
}

/// Two design ψ axes: a marginal length scale, then a slope length scale.
fn two_design_axis_blocks() -> Vec<Vec<CustomFamilyBlockPsiDerivative>> {
    let axis = |x_psi: Array2<f64>| {
        CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi,
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )
    };
    vec![
        Vec::new(),
        vec![axis(marginal_design_derivative())],
        vec![axis(slope_design_derivative())],
    ]
}

/// `D_β ∂²_ψψ' H` for design pairs: the marginal diagonal in both slope frames, and
/// the cross-block and slope-diagonal pairs of a time-constant slope.
#[test]
fn design_psi_pair_third_information_matches_finite_difference_2765() {
    let options = BlockwiseFitOptions::default();
    let blocks = two_design_axis_blocks();
    for (frame, psi_i, psi_j) in [
        (SlopeFrame::Static, 0, 0),
        (SlopeFrame::Static, 0, 1),
        (SlopeFrame::Static, 1, 1),
        (SlopeFrame::FollowUpVarying, 0, 0),
    ] {
        let (family, beta) = drift_family_and_states(frame);
        let total = beta.len();
        let analytic = family
            .design_psi_pair_hessian_directional_derivative_all_beta_axes_with_options(
                &states_at_beta(&family, &beta),
                &blocks,
                psi_i,
                psi_j,
                &options,
            )
            .expect("design-pair third information derivative")
            .expect("a design pair on supported blocks publishes its third information derivative");
        grade_all_beta_axes(
            &format!("{}/design pair ({psi_i},{psi_j})", frame.label()),
            &analytic,
            &beta,
            |displaced| {
                let terms = family
                    .psi_second_order_terms_inner_with_options(
                        &states_at_beta(&family, displaced),
                        &blocks,
                        psi_i,
                        psi_j,
                        None,
                        &options,
                    )
                    .expect("design pair terms")
                    .expect("a design pair on supported blocks publishes its terms");
                match terms.hessian_psi_psi_operator.as_ref() {
                    Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                    None => terms.hessian_psi_psi.clone(),
                }
            },
        );
    }
}

/// A symmetric contraction weight with every entry nonzero, so each cross term is graded.
fn contraction_weight(dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |(a, b)| {
        (0.37 * (a + b + 1) as f64).sin() + if a == b { 0.8 } else { 0.0 }
    })
}

/// gam#2930: the one-pass `⟨W, ∂_ψ H²[e_a, e_b]⟩` along a design ψ is the contraction with `W`
/// of the column-by-column `{D_β_a D_β ∂_ψ H[e_b]}`, to roundoff, on the marginal and the slope
/// surface of a time-constant slope.
#[test]
fn design_contracted_trace_hessian_psi_matches_column_contraction_2930() {
    let options = BlockwiseFitOptions::default();
    for axis in [PsiAxis::MarginalDesign, PsiAxis::SlopeDesign] {
        let layout = hyper_layout(axis);
        let (family, beta) = drift_family_and_states(SlopeFrame::Static);
        let states = states_at_beta(&family, &beta);
        let total = beta.len();
        let weight = contraction_weight(total);
        assert!(
            family
                .design_contracted_trace_hessian_psi_available(
                    &states,
                    layout.design_derivative_blocks(),
                    0
                )
                .expect("availability query"),
            "{axis:?}: a design ψ axis on the rigid frame must serve the one-pass contraction"
        );
        let contracted = family
            .design_contracted_trace_hessian_psi_with_options(
                &states,
                layout.design_derivative_blocks(),
                0,
                &weight,
                &options,
            )
            .expect("one-pass design contraction")
            .expect("a design ψ axis on the rigid frame serves the one-pass contraction");
        let mut reference = Array2::<f64>::zeros((total, total));
        for b in 0..total {
            let mut unit = Array1::<f64>::zeros(total);
            unit[b] = 1.0;
            let columns = family
                .design_psi_hessian_second_directional_derivative_all_beta_axes_with_options(
                    &states,
                    layout.design_derivative_blocks(),
                    0,
                    &unit,
                    &options,
                )
                .expect("column-by-column design third information derivative")
                .expect("a design ψ axis publishes its third information derivative");
            for a in 0..total {
                reference[[a, b]] = (&weight * &columns[a]).sum();
            }
        }
        let scale = max_abs(reference.iter());
        assert!(
            scale > 1e-8,
            "{axis:?}: the reference contraction is identically zero, so it grades nothing"
        );
        let gap = max_abs((&contracted - &reference).iter());
        assert!(
            gap <= 1e-12 * scale,
            "{axis:?}: one-pass {contracted:?} against column contraction {reference:?}"
        );
    }
}

/// gam#2930: `⟨W, ∂_ψ H³[v, e_a, e_b]⟩` along a design ψ is the coefficient derivative of the
/// one-pass `⟨W, ∂_ψ H²⟩` at fixed `W`, graded along every coefficient axis.
#[test]
fn design_contracted_trace_hessian_psi_directional_matches_finite_difference_2930() {
    let options = BlockwiseFitOptions::default();
    for axis in [PsiAxis::MarginalDesign, PsiAxis::SlopeDesign] {
        let layout = hyper_layout(axis);
        let (family, beta) = drift_family_and_states(SlopeFrame::Static);
        let states = states_at_beta(&family, &beta);
        let total = beta.len();
        let weight = contraction_weight(total);
        let analytic: Vec<Array2<f64>> = (0..total)
            .map(|coefficient_axis| {
                let mut unit = Array1::<f64>::zeros(total);
                unit[coefficient_axis] = 1.0;
                family
                    .design_contracted_trace_hessian_psi_directional_with_options(
                        &states,
                        layout.design_derivative_blocks(),
                        0,
                        &weight,
                        &unit,
                        &options,
                    )
                    .expect("directional design contraction")
                    .expect("a design ψ axis on the rigid frame serves the directional contraction")
            })
            .collect();
        grade_all_beta_axes_on_derived_band(
            &format!("static/{axis:?} contracted trace Hessian by beta"),
            &analytic,
            &beta,
            |displaced| {
                family
                    .design_contracted_trace_hessian_psi_with_options(
                        &states_at_beta(&family, displaced),
                        layout.design_derivative_blocks(),
                        0,
                        &weight,
                        &options,
                    )
                    .expect("displaced design contraction")
                    .expect("a design ψ axis on the rigid frame serves the one-pass contraction")
            },
        );
    }
}

/// gam#2930: `⟨W, ∂²_ψψ' H²[e_a, e_b]⟩` for design pairs is the coefficient derivative along
/// `e_b` of the contraction with `W` of `{D_β_a ∂²_ψψ' H}`, on the diagonal and cross-block pairs
/// of a time-constant slope.
#[test]
fn design_contracted_trace_hessian_psi_pair_matches_finite_difference_2930() {
    let options = BlockwiseFitOptions::default();
    let blocks = two_design_axis_blocks();
    for (psi_i, psi_j) in [(0, 0), (0, 1), (1, 1)] {
        let (family, beta) = drift_family_and_states(SlopeFrame::Static);
        let total = beta.len();
        let weight = contraction_weight(total);
        let analytic = family
            .design_contracted_trace_hessian_psi_pair_with_options(
                &states_at_beta(&family, &beta),
                &blocks,
                psi_i,
                psi_j,
                &weight,
                &options,
            )
            .expect("pair design contraction")
            .expect("a design pair on the rigid frame serves the pair contraction");
        let contracted_at = |displaced: &Array1<f64>| -> Array1<f64> {
            let axes = family
                .design_psi_pair_hessian_directional_derivative_all_beta_axes_with_options(
                    &states_at_beta(&family, displaced),
                    &blocks,
                    psi_i,
                    psi_j,
                    &options,
                )
                .expect("design-pair third information derivative")
                .expect("a design pair on supported blocks publishes its third information derivative");
            Array1::from_iter(axes.iter().map(|matrix| (&weight * matrix).sum()))
        };
        grade_contraction_by_beta(
            &format!("static/design pair ({psi_i},{psi_j}) contracted trace Hessian"),
            &analytic,
            &beta,
            contracted_at,
        );
    }
}

/// gam#2945: the graded step `g` of [`derived_difference`]. The stencil reaches `±2g = ±1e-3`.
const DERIVED_DIFFERENCE_STEP: f64 = 5e-4;

/// gam#2945: `f` on the stencil [`derived_difference`] reads, `[f(−2g), f(−g), f(g), f(2g)]`.
fn derived_difference_stencil<T>(f: impl Fn(f64) -> T) -> [T; 4] {
    let g = DERIVED_DIFFERENCE_STEP;
    [f(-2.0 * g), f(-g), f(g), f(2.0 * g)]
}

/// gam#2945: a central difference `D(g)` and an error band derived from the difference's own error
/// sources, not a literal floor.
///
/// Truncation: with `E(g) = D(g) − f′`, the disagreement `Δ₁ = D(2g) − D(g)` is `E(2g) − E(g)`, so
/// `|E(g)| ≤ |Δ₁|` whenever the error at least halves per halving, `|E(2g)| ≥ 2·|E(g)|`. A central
/// difference in its asymptotic range quarters it (`E = a·g² + …`), where `|E(g)| = |Δ₁|/3`. The richer
/// estimate `Δ₁/3 − (Δ₂ − 4Δ₁)/45` is not a bound: its correction can cancel while sixth-order error
/// remains, and it fell short by 0.04 % on a design-axis contraction here.
///
/// Roundoff: an object assembled as a floating sum of `n` summands carries Higham's recursive-summation
/// error `(n − 1)·ε·Σ|x|` beside the summands' own `ε·Σ|x|`, so `δ ≤ n·ε·Σ|x|`. The summands are not
/// published, so `Σ|x|` is taken at `M`, the largest magnitude the stencil took: exact when the summands
/// share a sign, a named proxy otherwise. With `δ = n·ε·M`, `D(g)` carries `δ/g` and `Δ₁` carries
/// `1.5·δ/g`, so the band adds `2.5·δ/g`.
///
/// The step `g` places the stencil; it is not a tolerance. The band is derived for whatever step is
/// used, so a different `g` moves the band with it and never loosens the gate.
#[derive(Clone, Copy, Debug)]
struct DerivedDifference {
    value: f64,
    band: f64,
}

fn derived_difference(stencil: [f64; 4], g: f64, summands: usize) -> DerivedDifference {
    let [minus_2, minus_1, plus_1, plus_2] = stencil;
    let d_1 = (plus_1 - minus_1) / (2.0 * g);
    let d_2 = (plus_2 - minus_2) / (4.0 * g);
    let magnitude = stencil.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let roundoff = summands as f64 * f64::EPSILON * magnitude;
    DerivedDifference {
        value: d_1,
        band: (d_2 - d_1).abs() + 2.5 * roundoff / g,
    }
}

/// gam#2945: grade `analytic` against a [`DerivedDifference`]. The difference must resolve the object
/// it grades, `band ≤ scale`, before any gap is charged to the analytic term.
fn assert_within_derived_band(label: &str, analytic: f64, oracle: DerivedDifference, scale: f64) {
    assert!(
        oracle.band <= scale,
        "{label}: the central difference did not resolve this object (band {:.3e} against scale \
         {scale:.3e}); the analytic term is not on trial here",
        oracle.band,
    );
    let gap = (analytic - oracle.value).abs();
    assert!(
        gap <= oracle.band,
        "{label}: analytic={analytic:.9e} fd={:.9e} gap={gap:.3e} band={:.3e}",
        oracle.value,
        oracle.band,
    );
}

/// gam#2945: [`grade_all_beta_axes`] on the band of [`derived_difference`]. Every object of the family
/// is resolved against the family's largest analytic magnitude. The object differenced is a one-pass
/// contraction `⟨W, ·⟩` over `N_ROWS` rows and the `p²` entries of `W`, so it carries `N_ROWS·p²`
/// summands.
fn grade_all_beta_axes_on_derived_band(
    label: &str,
    analytic: &[Array2<f64>],
    beta: &Array1<f64>,
    lower_order_at: impl Fn(&Array1<f64>) -> Array2<f64>,
) {
    let dim = beta.len();
    assert_eq!(analytic.len(), dim, "{label}: one matrix per coefficient axis");
    let scale = analytic
        .iter()
        .fold(0.0_f64, |acc, matrix| acc.max(max_abs(matrix.iter())));
    assert!(
        scale > 1e-8,
        "{label}: every analytic matrix is identically zero, so it grades nothing"
    );
    for (coefficient_axis, matrix) in analytic.iter().enumerate() {
        assert_eq!(matrix.dim(), (dim, dim), "{label}: axis {coefficient_axis} shape");
        let stencil = derived_difference_stencil(|t| {
            let mut displaced = beta.clone();
            displaced[coefficient_axis] += t;
            lower_order_at(&displaced)
        });
        for row in 0..dim {
            for column in 0..dim {
                assert_within_derived_band(
                    &format!("{label} coefficient axis {coefficient_axis} [{row},{column}]"),
                    matrix[[row, column]],
                    derived_difference(
                        std::array::from_fn(|k| stencil[k][[row, column]]),
                        DERIVED_DIFFERENCE_STEP,
                        N_ROWS * dim * dim,
                    ),
                    scale,
                );
            }
        }
    }
}

/// Grade `analytic[[a, b]]` against the central difference along `e_b` of `contracted_at(β)[a]`,
/// the contraction with `W` of each coefficient axis's lower-order derivative, on the band of
/// [`derived_difference`].
fn grade_contraction_by_beta(
    label: &str,
    analytic: &Array2<f64>,
    beta: &Array1<f64>,
    contracted_at: impl Fn(&Array1<f64>) -> Array1<f64>,
) {
    let total = beta.len();
    assert_eq!(analytic.dim(), (total, total), "{label}: shape");
    let scale = max_abs(analytic.iter());
    assert!(
        scale > 1e-8,
        "{label}: the contraction is identically zero, so it grades nothing"
    );
    for column in 0..total {
        let stencil = derived_difference_stencil(|t| {
            let mut displaced = beta.clone();
            displaced[column] += t;
            contracted_at(&displaced)
        });
        for row in 0..total {
            assert_within_derived_band(
                &format!("{label} [{row},{column}]"),
                analytic[[row, column]],
                derived_difference(
                    std::array::from_fn(|k| stencil[k][row]),
                    DERIVED_DIFFERENCE_STEP,
                    N_ROWS * total * total,
                ),
                scale,
            );
        }
    }
}

/// gam#2930: along a baseline-chart coordinate, `⟨W, ∂_θ H³[v, e_a, e_b]⟩` is the coefficient
/// derivative of the one-pass `⟨W, ∂_θ H²⟩` at fixed `W`, graded along every coefficient axis.
#[test]
fn baseline_contracted_trace_hessian_psi_directional_matches_finite_difference_2930() {
    let options = BlockwiseFitOptions::default();
    let (family, beta) = drift_family_and_states(SlopeFrame::Static);
    let states = states_at_beta(&family, &beta);
    let total = beta.len();
    let weight = contraction_weight(total);
    for axis in [0, 1, 2] {
        let analytic: Vec<Array2<f64>> = (0..total)
            .map(|coefficient_axis| {
                let mut unit = Array1::<f64>::zeros(total);
                unit[coefficient_axis] = 1.0;
                family
                    .baseline_contracted_trace_hessian_psi_directional_with_options(
                        &states, axis, &weight, &unit, &options,
                    )
                    .expect("directional chart contraction")
                    .expect("a chart axis on the rigid frame serves the directional contraction")
            })
            .collect();
        grade_all_beta_axes_on_derived_band(
            &format!("static/chart axis {axis} contracted trace Hessian by beta"),
            &analytic,
            &beta,
            |displaced| {
                family
                    .baseline_contracted_trace_hessian_psi_with_options(
                        &states_at_beta(&family, displaced),
                        axis,
                        &weight,
                        &options,
                    )
                    .expect("displaced chart contraction")
                    .expect("a chart axis on the rigid frame serves the one-pass contraction")
            },
        );
    }
}

/// gam#2930: `⟨W, ∂²_θθ' H²[e_a, e_b]⟩` for baseline-chart pairs is the coefficient derivative
/// along `e_b` of the contraction with `W` of `{D_β_a ∂²_θθ' H}`, on diagonal and cross pairs.
#[test]
fn baseline_contracted_trace_hessian_psi_pair_matches_finite_difference_2930() {
    let options = BlockwiseFitOptions::default();
    let (family, beta) = drift_family_and_states(SlopeFrame::Static);
    let weight = contraction_weight(beta.len());
    for (axis, other_axis) in [(0, 0), (1, 1), (0, 2), (1, 2)] {
        let analytic = family
            .baseline_contracted_trace_hessian_psi_pair_with_options(
                &states_at_beta(&family, &beta),
                axis,
                other_axis,
                &weight,
                &options,
            )
            .expect("pair chart contraction")
            .expect("a chart pair on the rigid frame serves the pair contraction");
        grade_contraction_by_beta(
            &format!("static/chart pair ({axis},{other_axis}) contracted trace Hessian"),
            &analytic,
            &beta,
            |displaced| {
                let axes = family
                    .baseline_exact_joint_psisecond_order_hessian_directional_derivative_all_beta_axes_with_options(
                        &states_at_beta(&family, displaced),
                        axis,
                        other_axis,
                        &options,
                    )
                    .expect("chart-pair third information derivative");
                Array1::from_iter(axes.iter().map(|matrix| (&weight * matrix).sum()))
            },
        );
    }
}

/// gam#2945 positive control for [`baseline_contracted_trace_hessian_psi_pair_matches_finite_difference_2930`]:
/// the same derived band ([`derived_difference`]) rejects the pair contraction with the chart's second
/// primary shift `∂²_θθ' (q₀, q₁, q̇₁)` zeroed. That shift is the `T5[d_θθ']` half of
/// `⟨W, ∂²_θθ' H²⟩`; a pair calculus that omits it must not pass the gate.
#[test]
fn baseline_psi_pair_gate_rejects_a_dropped_second_chart_shift_2930() {
    use super::row_kernel::SurvivalMarginalSlopeRowKernel;
    use super::slope_geometry::{STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry};

    let options = BlockwiseFitOptions::default();
    let (family, beta) = drift_family_and_states(SlopeFrame::Static);
    let weight = contraction_weight(beta.len());
    let states = states_at_beta(&family, &beta);
    let geometry = family
        .rigid_baseline_geometry()
        .expect("the chart fixture carries frozen baseline geometry");
    let row_weights = family.rigid_third_row_weights(&options);
    let mut worst = 0.0_f64;
    for (axis, other_axis) in [(0, 0), (1, 1), (0, 2), (1, 2)] {
        let dropped =
            SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
                family.clone(),
                states.clone(),
            )
            .contracted_trace_hessian_primary_shift_pair(&weight, &row_weights, |row| {
                Ok((
                    SurvivalMarginalSlopeFamily::rigid_baseline_primary_first::<
                        STATIC_SLOPE_PRIMARIES,
                    >(geometry, row, axis)?,
                    SurvivalMarginalSlopeFamily::rigid_baseline_primary_first::<
                        STATIC_SLOPE_PRIMARIES,
                    >(geometry, row, other_axis)?,
                    [0.0; STATIC_SLOPE_PRIMARIES],
                ))
            })
            .expect("pair contraction with the second chart shift dropped");
        let contracted_at = |displaced: &Array1<f64>| {
            let axes = family
                .baseline_exact_joint_psisecond_order_hessian_directional_derivative_all_beta_axes_with_options(
                    &states_at_beta(&family, displaced),
                    axis,
                    other_axis,
                    &options,
                )
                .expect("chart-pair third information derivative");
            Array1::from_iter(axes.iter().map(|matrix| (&weight * matrix).sum()))
        };
        let scale = max_abs(dropped.iter());
        for column in 0..beta.len() {
            let stencil = derived_difference_stencil(|t| {
                let mut displaced = beta.clone();
                displaced[column] += t;
                contracted_at(&displaced)
            });
            for row in 0..beta.len() {
                let oracle = derived_difference(
                    std::array::from_fn(|k| stencil[k][row]),
                    DERIVED_DIFFERENCE_STEP,
                    N_ROWS * beta.len() * beta.len(),
                );
                if oracle.band > scale {
                    continue;
                }
                worst = worst.max((dropped[[row, column]] - oracle.value).abs() / oracle.band);
            }
        }
    }
    eprintln!("[2930-CONTROL] second chart shift dropped: worst gap over the gate band {worst:.3e}");
    assert!(
        worst > 1.0,
        "the pair gate accepted a contraction missing the chart's second primary shift \
         (worst gap over band {worst:.3e}), so it grades nothing about that term"
    );
}

/// `D_β ∂²_θθ' H` for diagonal and cross baseline pairs, in both slope frames.
#[test]
fn baseline_psi_pair_third_information_matches_finite_difference_2765() {
    let options = BlockwiseFitOptions::default();
    for frame in [SlopeFrame::Static, SlopeFrame::FollowUpVarying] {
        let (family, beta) = drift_family_and_states(frame);
        let total = beta.len();
        for (axis, other_axis) in [(0, 0), (1, 1), (0, 2), (1, 2)] {
            let analytic = family
                .baseline_exact_joint_psisecond_order_hessian_directional_derivative_all_beta_axes_with_options(
                    &states_at_beta(&family, &beta),
                    axis,
                    other_axis,
                    &options,
                )
                .expect("baseline-pair third information derivative");
            grade_all_beta_axes(
                &format!("{}/baseline pair ({axis},{other_axis})", frame.label()),
                &analytic,
                &beta,
                |displaced| {
                    let terms = family
                        .baseline_exact_joint_psisecond_order_terms_with_options(
                            &states_at_beta(&family, displaced),
                            axis,
                            other_axis,
                            &options,
                        )
                        .expect("baseline pair terms")
                        .expect("a rigid baseline chart publishes its pair terms");
                    match terms.hessian_psi_psi_operator.as_ref() {
                        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                        None => terms.hessian_psi_psi.clone(),
                    }
                },
            );
        }
    }
}
