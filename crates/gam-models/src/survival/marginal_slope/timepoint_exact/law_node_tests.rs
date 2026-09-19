//! gam#2948: the flex row program anchored on a declared finite law.
//!
//! A score warp or a link deviation must not move the anchor
//! `Σ_k w_k Φ(−η_k) = Φ(−q)`: each timepoint's intercept is solved on the
//! family's own law, and every jet differentiates that solve. Three pins:
//!
//! 1. **The anchor.** On a skewed law the Gaussian program's intercept misses
//!    the law's marginal identity, and a warp or deviation moves the miss. The
//!    anchored program's intercept meets the identity for every coefficient
//!    vector.
//! 2. **The base anchor.** With zero warp and deviation coefficients the
//!    anchored flex program is the rigid anchored frame of gam#2923, an
//!    independent implementation of the same identity.
//! 3. **Finite differences.** The anchored row's gradient, Hessian, contracted
//!    third and contracted fourth are central differences of its value,
//!    gradient, Hessian and Hessian.

use super::*;
use crate::bms::{EmpiricalZGrid, LatentMeasureKind};
use crate::test_support::skewed_grid;
use ndarray::array;

const N: usize = 16;
const ROWS: [usize; 3] = [2, 6, 11];

fn deviation_runtime() -> DeviationRuntime {
    build_score_warp_deviation_block_from_seed(
        &Array1::from(vec![-1.0, 0.0, 1.0]),
        &DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 1,
            penalty_order: 2,
            penalty_orders: vec![1, 2, 3],
            double_penalty: false,
            monotonicity_eps: 1e-4,
        },
    )
    .expect("build a deviation runtime")
    .runtime
}

/// A scalar-score survival family with a score warp and a link deviation on the
/// Gaussian law: deterministic rows, one-column marginal and slope designs, and no
/// time design, so the offsets carry the baseline.
fn gaussian_flex_family() -> SurvivalMarginalSlopeFamily {
    let n = N;
    let unit = |i: usize, a: usize, b: usize| ((i * a + b) % n) as f64 + 0.5;
    let event = Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights = Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z = Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * unit(i, 17, 5) / n as f64));
    let offset_entry = Array1::from_iter((0..n).map(|i| -0.4 + 0.7 * unit(i, 11, 3) / n as f64));
    let offset_exit = Array1::from_iter((0..n).map(|i| 0.1 + 0.6 * unit(i, 19, 7) / n as f64));
    let derivative_offset_exit =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    let marginal_design =
        Array2::from_shape_fn((n, 1), |(i, _)| 0.3 + 0.4 * ((i * 29 + 11) % n) as f64 / n as f64);
    let slope_design =
        Array2::from_shape_fn((n, 1), |(i, _)| 0.2 + 0.5 * ((i * 37 + 9) % n) as f64 / n as f64);
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: MarginalSlopeCovariance::diagonal(array![1.0])
            .expect("unit score covariance")
            .into(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        marginal_design: DesignMatrix::from(marginal_design),
        slope_layout: DesignMatrix::from(slope_design).into(),
        score_warp: Some(deviation_runtime()),
        link_dev: Some(deviation_runtime()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
    }
}

/// `family` anchored on the finite law `grid`.
fn anchored_on(
    family: &SurvivalMarginalSlopeFamily,
    grid: &AnchorGridOwned,
) -> SurvivalMarginalSlopeFamily {
    let grid = EmpiricalZGrid::new(grid.nodes.clone(), grid.weights.clone(), "gam#2948 test law")
        .expect("a valid finite law");
    let law = SurvivalLatentLaw::from_kind(&LatentMeasureKind::GlobalEmpirical { grid }, family.n)
        .expect("materialise the law")
        .expect("a finite law is not the standard-normal law");
    SurvivalMarginalSlopeFamily {
        latent_law: Some(Arc::new(law)),
        ..family.clone()
    }
}

/// Warp and deviation coefficients at `amplitude` times a fixed pattern.
fn warp_and_deviation(
    family: &SurvivalMarginalSlopeFamily,
    amplitude: f64,
) -> (Array1<f64>, Array1<f64>) {
    let warp = family.score_warp.as_ref().expect("score warp").basis_dim();
    let deviation = family.link_dev.as_ref().expect("link deviation").basis_dim();
    (
        Array1::from_iter(
            (0..warp).map(|i| amplitude * (0.1 + 0.05 * i as f64 - 0.02 * (i % 2) as f64)),
        ),
        Array1::from_iter(
            (0..deviation).map(|i| amplitude * (-0.08 + 0.04 * i as f64 + 0.01 * (i % 3) as f64)),
        ),
    )
}

fn block_states(
    family: &SurvivalMarginalSlopeFamily,
    slope_beta: f64,
    beta_h: &Array1<f64>,
    beta_w: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let marginal = family.marginal_design.to_dense().to_owned();
    let slope = family.slope_layout.coefficient_design().to_dense().to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: array![0.15],
            eta: marginal.dot(&array![0.15]),
        },
        ParameterBlockState {
            beta: array![slope_beta],
            eta: slope.dot(&array![slope_beta]),
        },
        ParameterBlockState {
            beta: beta_h.clone(),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: beta_w.clone(),
            eta: Array1::zeros(n),
        },
    ]
}

fn assert_relatively_close(label: &str, analytic: f64, reference: f64, tolerance: f64) {
    assert!(
        (analytic - reference).abs() <= tolerance * (1.0 + reference.abs()),
        "{label}: {analytic:.12e} vs {reference:.12e} (tolerance {tolerance:.1e})"
    );
}

/// On a skewed law the Gaussian flex program's intercept misses the law's
/// marginal identity by an amount the warp and the deviation move; the anchored
/// program's intercept meets it for every coefficient vector. The identity is
/// read at each node through the shared de-nested cell kernel, not through the
/// node sum the anchored solve evaluates.
#[test]
fn flex_blocks_cannot_move_the_anchor_on_a_declared_law_2948() {
    let gaussian = gaussian_flex_family();
    let law = skewed_grid();
    let anchored = anchored_on(&gaussian, &law);
    let scale = gaussian.probit_frailty_scale();
    // The identity on the smaller tail, in log units (gam#2971): `T = Σ w Φ(−η)`
    // where `q ≥ 0` and `T = Σ w Φ(η)` otherwise, against `Φ(∓q)`. Returns the
    // residual `log T − log Φ(∓q)` and the certificate bound the production solve
    // enforces at that intercept.
    let identity_residual = |a: f64, q: f64, g: f64, beta_h: &Array1<f64>, beta_w: &Array1<f64>| {
        let survival_side = q >= 0.0;
        let mut tail = 0.0;
        let mut density = 0.0;
        for (&u, &w) in law.nodes.iter().zip(&law.weights) {
            let partials = shared_observed_denested_cell_partials(
                u,
                a,
                g,
                gaussian.score_warp.as_ref(),
                Some(beta_h),
                gaussian.link_dev.as_ref(),
                Some(beta_w),
                scale,
            )
            .expect("node partials");
            let eta = eval_coeff4_at(&partials.coeff, u);
            let chi = eval_coeff4_at(&partials.dc_da, u);
            tail += w * crate::probability::normal_cdf(if survival_side { -eta } else { eta });
            density += w * crate::probability::normal_pdf(eta) * chi;
        }
        let log_target = if survival_side {
            crate::probability::normal_logcdf(-q)
        } else {
            crate::probability::normal_logcdf(q)
        };
        let log_slope = (density / tail).abs();
        let rounding = crate::latent_anchor::anchor_residual_rounding(log_target, law.nodes.len());
        let bound = crate::latent_anchor::anchor_residual_resolution(a, log_slope, rounding);
        (tail.ln() - log_target, bound)
    };
    let slope = 0.35;
    let (flat_h, flat_w) = warp_and_deviation(&gaussian, 0.0);
    let (warp_h, warp_w) = warp_and_deviation(&gaussian, 2.0);
    let mut unresolved: Vec<String> = Vec::new();
    let mut unmoved: Vec<String> = Vec::new();
    for q in [-1.2, -0.3, 0.4, 1.1] {
        let solve = |family: &SurvivalMarginalSlopeFamily, beta_h: &Array1<f64>, beta_w: &Array1<f64>| {
            family
                .solve_row_survival_intercept_with_slot(q, slope, Some(beta_h), Some(beta_w), None)
                .expect("intercept solve")
                .0
        };
        let (gaussian_flat, gaussian_flat_bound) =
            identity_residual(solve(&gaussian, &flat_h, &flat_w), q, slope, &flat_h, &flat_w);
        let (gaussian_warped, gaussian_warped_bound) =
            identity_residual(solve(&gaussian, &warp_h, &warp_w), q, slope, &warp_h, &warp_w);
        let (anchored_flat, anchored_flat_bound) =
            identity_residual(solve(&anchored, &flat_h, &flat_w), q, slope, &flat_h, &flat_w);
        let (anchored_warped, anchored_warped_bound) =
            identity_residual(solve(&anchored, &warp_h, &warp_w), q, slope, &warp_h, &warp_w);
        let gaussian_move = (gaussian_warped - gaussian_flat).abs();
        let resolvable = anchored_flat_bound + anchored_warped_bound;
        eprintln!(
            "[2948 anchor] q={q:+.2} gaussian log miss flat={gaussian_flat:+.3e} (bound {gaussian_flat_bound:.3e}) \
             warped={gaussian_warped:+.3e} (bound {gaussian_warped_bound:.3e}) move={gaussian_move:.3e} | \
             anchored log miss flat={anchored_flat:+.3e} (bound {anchored_flat_bound:.3e}) \
             warped={anchored_warped:+.3e} (bound {anchored_warped_bound:.3e})"
        );
        if anchored_flat.abs() > anchored_flat_bound {
            unresolved.push(format!("q={q:+.2} flat {anchored_flat:+.3e} > {anchored_flat_bound:.3e}"));
        }
        if anchored_warped.abs() > anchored_warped_bound {
            unresolved.push(format!(
                "q={q:+.2} warped {anchored_warped:+.3e} > {anchored_warped_bound:.3e}"
            ));
        }
        if !(gaussian_move > resolvable) {
            unmoved.push(format!("q={q:+.2} move {gaussian_move:.3e} <= {resolvable:.3e}"));
        }
    }
    // The anchored solve certifies its log-tail residual at the resolution the
    // solve itself enforces, read here through the independent node kernel.
    assert!(
        unresolved.is_empty(),
        "the anchored flex intercept must meet the declared law's marginal identity to its \
         certificate for every warp and deviation: {unresolved:?}"
    );
    // The Gaussian program's anchor must move with the warp and the deviation by
    // more than the anchored certificate could hide, or the pin bounds nothing.
    assert!(
        unmoved.is_empty(),
        "the fixture must show the Gaussian program's log miss moving with the warp and the \
         deviation by more than the anchored certificate resolves: {unmoved:?}"
    );
}

/// With zero warp and deviation coefficients the anchored flex program anchors
/// where the rigid anchored frame does, and its row value, gradient and Hessian
/// in the four core primaries are that frame's. The two share no code below the
/// law itself: the flex program solves the node sum through its cubic spans and
/// lifts the intercept in its jet algebra, the rigid frame reads the anchor's
/// implicit-derivative table.
#[test]
fn zero_deviation_anchored_flex_is_the_rigid_anchored_frame_2948() {
    let law = skewed_grid();
    let flex = anchored_on(&gaussian_flex_family(), &law);
    let rigid = SurvivalMarginalSlopeFamily {
        score_warp: None,
        link_dev: None,
        ..flex.clone()
    };
    let (beta_h, beta_w) = warp_and_deviation(&flex, 0.0);
    let flex_states = block_states(&flex, 0.25, &beta_h, &beta_w);
    let rigid_states = flex_states[..3].to_vec();
    let primary = flex_primary_slices(&flex);
    for row in ROWS {
        let q = flex.row_dynamic_q_values(row, &flex_states).expect("row q");
        let slope = flex_states[2].eta[row];
        let (flex_value, flex_gradient, flex_hessian) = flex
            .compute_row_flex_primary_gradient_hessian_from_parts(
                row,
                q.q0,
                q.q1,
                q.qd1,
                slope,
                Some(&beta_h),
                Some(&beta_w),
                0.0,
                &primary,
            )
            .expect("anchored flex row");
        let (rigid_value, rigid_gradient, rigid_hessian) = rigid
            .compute_row_primary_gradient_hessian_uncached(row, &rigid_states)
            .expect("rigid anchored row");
        eprintln!(
            "[2948 base anchor] row {row}: value flex={flex_value:.12e} rigid={rigid_value:.12e}"
        );
        assert_relatively_close(&format!("row {row} value"), flex_value, rigid_value, 1e-8);
        let core = [primary.q0, primary.q1, primary.qd1, primary.g];
        for (i, &axis) in core.iter().enumerate() {
            assert_relatively_close(
                &format!("row {row} gradient[{i}]"),
                flex_gradient[axis],
                rigid_gradient[i],
                1e-8,
            );
            for (j, &other) in core.iter().enumerate() {
                assert_relatively_close(
                    &format!("row {row} Hessian[{i},{j}]"),
                    flex_hessian[[axis, other]],
                    rigid_hessian[[i, j]],
                    1e-8,
                );
            }
        }
    }
}

/// Central differences of the anchored flex row in every primary: of the value
/// for the gradient, of the gradient for the Hessian, of the Hessian along one
/// direction for the contracted third, and a mixed second difference of the
/// Hessian for the contracted fourth. Every stencil point re-solves both
/// intercepts on the declared law.
#[test]
fn anchored_flex_row_matches_finite_differences_2948() {
    let law = skewed_grid();
    let family = anchored_on(&gaussian_flex_family(), &law);
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let h_range = primary.h.clone().expect("warp primaries");
    let w_range = primary.w.clone().expect("deviation primaries");
    let (beta_h, beta_w) = warp_and_deviation(&family, 1.0);
    let states = block_states(&family, 0.25, &beta_h, &beta_w);
    for row in ROWS {
        let q = family.row_dynamic_q_values(row, &states).expect("row q");
        let mut theta = Array1::<f64>::zeros(p);
        theta[primary.q0] = q.q0;
        theta[primary.q1] = q.q1;
        theta[primary.qd1] = q.qd1;
        theta[primary.g] = states[2].eta[row];
        theta.slice_mut(s![h_range.clone()]).assign(&beta_h);
        theta.slice_mut(s![w_range.clone()]).assign(&beta_w);
        let coefficients = |theta: &Array1<f64>| {
            (
                theta.slice(s![h_range.clone()]).to_owned(),
                theta.slice(s![w_range.clone()]).to_owned(),
            )
        };
        let value = |theta: &Array1<f64>| -> f64 {
            let (bh, bw) = coefficients(theta);
            family
                .row_neglog_flex_value_from_parts(
                    row,
                    theta[primary.q0],
                    theta[primary.q1],
                    theta[primary.qd1],
                    theta[primary.g],
                    Some(&bh),
                    Some(&bw),
                    0.0,
                )
                .expect("row value")
        };
        let parts = |theta: &Array1<f64>| -> (f64, Array1<f64>, Array2<f64>) {
            let (bh, bw) = coefficients(theta);
            family
                .compute_row_flex_primary_gradient_hessian_from_parts(
                    row,
                    theta[primary.q0],
                    theta[primary.q1],
                    theta[primary.qd1],
                    theta[primary.g],
                    Some(&bh),
                    Some(&bw),
                    0.0,
                    &primary,
                )
                .expect("row gradient and Hessian")
        };

        let (jet_value, gradient, hessian) = parts(&theta);
        assert_relatively_close(&format!("row {row} value"), jet_value, value(&theta), 1e-10);

        // Each value carries its intercepts' solve error, an absolute probability
        // residual of 1e-12, so a difference over 2e-5 resolves the gradient to
        // well inside 1e-5.
        let step = 1e-5;
        for axis in 0..p {
            let mut plus = theta.clone();
            plus[axis] += step;
            let mut minus = theta.clone();
            minus[axis] -= step;
            assert_relatively_close(
                &format!("row {row} gradient[{axis}]"),
                gradient[axis],
                (value(&plus) - value(&minus)) / (2.0 * step),
                1e-5,
            );
            let (_, gradient_plus, _) = parts(&plus);
            let (_, gradient_minus, _) = parts(&minus);
            for other in 0..p {
                assert_relatively_close(
                    &format!("row {row} Hessian[{axis},{other}]"),
                    hessian[[axis, other]],
                    (gradient_plus[other] - gradient_minus[other]) / (2.0 * step),
                    1e-5,
                );
            }
        }

        let dir_u = Array1::from_iter((0..p).map(|c| 0.12 + 0.04 * c as f64 - 0.01 * (c % 2) as f64));
        let dir_v = Array1::from_iter((0..p).map(|c| -0.07 + 0.05 * (c % 3) as f64 + 0.02 * c as f64));
        let third = family
            .row_flex_primary_third_contracted_exact(row, &states, &dir_u)
            .expect("contracted third");
        let t = 1e-4;
        let (_, _, hessian_plus) = parts(&(&theta + &(&dir_u * t)));
        let (_, _, hessian_minus) = parts(&(&theta - &(&dir_u * t)));
        let fourth = family
            .row_flex_primary_fourth_contracted_exact(row, &states, &dir_u, &dir_v)
            .expect("contracted fourth");
        // The mixed stencil's step matches the established flex fourth-order gate.
        let s = 2e-3;
        let displaced = |a: f64, b: f64| parts(&(&theta + &(&dir_u * a) + &(&dir_v * b))).2;
        let mixed = (displaced(s, s) - displaced(s, -s) - displaced(-s, s) + displaced(-s, -s))
            / (4.0 * s * s);
        for a in 0..p {
            for b in 0..p {
                assert_relatively_close(
                    &format!("row {row} third[{a},{b}]"),
                    third[[a, b]],
                    (hessian_plus[[a, b]] - hessian_minus[[a, b]]) / (2.0 * t),
                    1e-4,
                );
                assert_relatively_close(
                    &format!("row {row} fourth[{a},{b}]"),
                    fourth[[a, b]],
                    mixed[[a, b]],
                    1e-3,
                );
            }
        }
    }
}
