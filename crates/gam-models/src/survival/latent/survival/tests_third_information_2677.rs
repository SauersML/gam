#![cfg(test)]
//! #2677: the latent families' third information derivative.
//!
//! An armed Jeffreys term's exact outer Hessian reads `{D³H[u, v, e_a]}` through
//! `JeffreysThirdInformationDerivative`. Both latent families take their Jeffreys
//! information to be the observed joint Hessian, so the derivative is the fifth
//! likelihood derivative contracted with `(u, v)` and each coefficient axis.
//!
//! The row kernels are checked against a five-point difference of the production
//! contracted fourth along the third direction, on every event branch and at both
//! live primary dimensions. The assembled family derivatives are checked against
//! a five-point β-difference of the production `D²H[u, v]`.

use super::*;
use ndarray::{Array1, Array2, array, s};

const DIFFERENCE_STEP: f64 = 1.0e-3;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

/// The five-point differences `(−F(2h) + 8F(h) − 8F(−h) + F(−2h)) / 12h` of one
/// production derivative at `h` and at `2h`, with the largest stencil entry over both.
///
/// The production contracted derivatives read cumulants out of a moment table
/// through cancellation, so one evaluation carries far more rounding than
/// `ε·max|F|`, and the stencil amplifies it by `1/h`. The two steps' disagreement
/// measures the oracle's own resolution, truncation included. Job 1223327 measured
/// the exact-event row's log-σ channel: the estimates scatter by about ±3e-9 (2e-6
/// to 4e-6 relative) around the exact fifth for `h` from 4e-3 to 1e-3, alternating
/// in sign, and grow as `h` shrinks below that.
struct DifferenceOracle {
    fine: Array2<f64>,
    coarse: Array2<f64>,
    stencil_max: f64,
}

fn five_point_difference(evaluate: impl Fn(f64) -> Array2<f64>, h: f64) -> DifferenceOracle {
    let at_step = |step: f64| {
        let plus2 = evaluate(2.0 * step);
        let plus = evaluate(step);
        let minus = evaluate(-step);
        let minus2 = evaluate(-2.0 * step);
        let stencil_max = [&plus2, &plus, &minus, &minus2]
            .iter()
            .flat_map(|matrix| matrix.iter())
            .fold(0.0_f64, |worst, value| worst.max(value.abs()));
        (
            (&(&plus * 8.0) - &(&minus * 8.0) - &plus2 + &minus2) / (12.0 * step),
            stencil_max,
        )
    };
    let (fine, fine_max) = at_step(h);
    let (coarse, coarse_max) = at_step(2.0 * h);
    DifferenceOracle {
        fine,
        coarse,
        stencil_max: fine_max.max(coarse_max),
    }
}

/// The worst channel of `derivative` against the oracle, scaled by the bar:
/// `RELATIVE_TOLERANCE` of the larger magnitude, `ABSOLUTE_TOLERANCE`, and the
/// oracle's measured resolution `|D(h) − D(2h)|` at that channel.
fn worst_scaled_error(derivative: &Array2<f64>, oracle: &DifferenceOracle) -> (f64, (usize, usize)) {
    let mut worst = 0.0_f64;
    let mut worst_channel = (0, 0);
    for ((a, b), &left) in derivative.indexed_iter() {
        let right = oracle.fine[[a, b]];
        let resolution = (right - oracle.coarse[[a, b]]).abs();
        let scaled = (left - right).abs()
            / (RELATIVE_TOLERANCE * left.abs().max(right.abs()) + ABSOLUTE_TOLERANCE + resolution);
        if !scaled.is_finite() || scaled > worst {
            worst = if scaled.is_finite() { scaled } else { f64::INFINITY };
            worst_channel = (a, b);
        }
    }
    (worst, worst_channel)
}

/// The exact matrix agrees with its difference oracle channel by channel. The
/// stencil's rounding reaches `1.5·ε·max|F|/h`, and a relative bar of
/// `RELATIVE_TOLERANCE` is only meetable above that bound divided by the bar, so
/// the exact derivative must clear it for agreement to say anything.
fn assert_matches_difference(label: &str, exact: &Array2<f64>, oracle: &DifferenceOracle) {
    let magnitude = exact.iter().fold(0.0_f64, |worst, value| worst.max(value.abs()));
    let rounding = 1.5 * f64::EPSILON * oracle.stencil_max / DIFFERENCE_STEP;
    let floor = rounding / RELATIVE_TOLERANCE;
    assert!(
        magnitude > floor,
        "{label}: the exact derivative is at most {magnitude:e}, below the difference oracle's \
         resolution floor {floor:e} (stencil max {:e}), so agreement would prove nothing",
        oracle.stencil_max
    );
    let (worst, (a, b)) = worst_scaled_error(exact, oracle);
    assert!(
        worst <= 1.0,
        "{label}: exact derivative differs from its five-point difference: worst scaled error \
         {worst:e} at [{a},{b}] (exact={:.17e}, difference={:.17e}, resolution={:e}, \
         magnitude={magnitude:e}, rel_tol={RELATIVE_TOLERANCE:e}, abs_tol={ABSOLUTE_TOLERANCE:e}, \
         h={DIFFERENCE_STEP:e})",
        exact[[a, b]],
        oracle.fine[[a, b]],
        (oracle.fine[[a, b]] - oracle.coarse[[a, b]]).abs(),
    );
}

/// A wrong derivative must fail the same bar, so the oracle's measured resolution
/// cannot hide a real error.
fn assert_rejects_difference(label: &str, wrong: &Array2<f64>, oracle: &DifferenceOracle) {
    let (worst, (a, b)) = worst_scaled_error(wrong, oracle);
    assert!(
        worst > 1.0,
        "{label}: a wrong derivative passes the difference bar: worst scaled error {worst:e} at [{a},{b}]"
    );
}

fn row_point() -> LatentSurvivalPrimaryPoint {
    LatentSurvivalPrimaryPoint {
        q_entry: -1.2,
        q_exit: -0.4,
        qdot_exit: 0.73,
        q_right: 0.5,
        mu: -0.15,
        sigma: 0.3_f64.exp(),
    }
}

/// `point` moved by `t` along a primary direction, with the scale moving in `log σ`.
fn moved_point(point: LatentSurvivalPrimaryPoint, direction: &Array1<f64>, t: f64) -> LatentSurvivalPrimaryPoint {
    LatentSurvivalPrimaryPoint {
        q_entry: point.q_entry + t * direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
        q_exit: point.q_exit + t * direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
        qdot_exit: point.qdot_exit + t * direction[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
        q_right: point.q_right + t * direction[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
        mu: point.mu + t * direction[LATENT_SURVIVAL_PRIMARY_MU],
        sigma: point.sigma * (t * direction[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]).exp(),
    }
}

#[test]
fn latent_survival_row_fifth_is_the_directional_derivative_of_the_fourth_2677() {
    let quadctx = QuadratureContext::new();
    let rows = [
        ("right", LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02)),
        ("exact", LatentSurvivalRow::exact_event(0.3, 0.67, 0.01, 0.02, 0.73, 0.08)),
        (
            "interval",
            LatentSurvivalRow::interval_censored(0.3, 0.67, 1.65, 0.01, 0.02, 0.05),
        ),
    ];
    for (event, row) in &rows {
        for include_log_sigma in [false, true] {
            let scale = if include_log_sigma { 1.0 } else { 0.0 };
            let direction_u = array![0.17, -0.11, 0.09, 0.13, -0.07, 0.05 * scale];
            let direction_v = array![-0.08, 0.14, -0.06, 0.04, 0.12, -0.09 * scale];
            let direction_w = array![0.05, 0.10, -0.07, 0.08, 0.11, 0.06 * scale];
            let point = row_point();
            let exact = latent_survival_row_primary_fifth_contracted(
                &quadctx,
                row,
                point,
                &direction_u,
                &direction_v,
                &direction_w,
                include_log_sigma,
            )
            .expect("three-seed row fifth");
            let difference = five_point_difference(
                |t| {
                    latent_survival_row_primary_fourth_contracted(
                        &quadctx,
                        row,
                        moved_point(point, &direction_w, t),
                        &direction_u,
                        &direction_v,
                        include_log_sigma,
                    )
                    .expect("two-seed row fourth at a moved point")
                },
                DIFFERENCE_STEP,
            );
            assert_matches_difference(
                &format!("#2677 latent survival row fifth, event={event}, log_sigma={include_log_sigma}"),
                &exact,
                &difference,
            );
            let along_v = latent_survival_row_primary_fifth_contracted(
                &quadctx,
                row,
                point,
                &direction_u,
                &direction_v,
                &direction_v,
                include_log_sigma,
            )
            .expect("three-seed row fifth along v");
            assert_rejects_difference(
                &format!(
                    "#2677 control: latent survival row fifth along v against the difference along w, \
                     event={event}, log_sigma={include_log_sigma}"
                ),
                &along_v,
                &difference,
            );
        }
    }
}

#[test]
fn latent_binary_row_fifth_is_the_directional_derivative_of_the_fourth_2677() {
    let quadctx = QuadratureContext::new();
    let row = LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02);
    // The binary deployment reads only the entry, exit and mean primaries.
    let direction_u = array![0.17, -0.11, 0.0, 0.0, -0.07, 0.0];
    let direction_v = array![-0.08, 0.14, 0.0, 0.0, 0.12, 0.0];
    let direction_w = array![0.05, 0.10, 0.0, 0.0, 0.11, 0.0];
    let point = LatentSurvivalPrimaryPoint {
        q_entry: -1.2,
        q_exit: -0.4,
        qdot_exit: 1.0,
        q_right: -0.4,
        mu: -0.15,
        sigma: 0.35,
    };
    for event in [0u8, 1u8] {
        let exact = latent_binary_row_contracted_fifth(
            &quadctx,
            &row,
            point,
            event,
            &direction_u,
            &direction_v,
            &direction_w,
        )
        .expect("three-seed binary row fifth");
        let difference = five_point_difference(
            |t| {
                let moved = moved_point(point, &direction_w, t);
                latent_binary_row_contracted_fourth(
                    &quadctx,
                    &row,
                    LatentSurvivalPrimaryPoint {
                        q_right: moved.q_exit,
                        ..moved
                    },
                    event,
                    &direction_u,
                    &direction_v,
                )
                .expect("two-seed binary row fourth at a moved point")
            },
            DIFFERENCE_STEP,
        );
        assert_matches_difference(
            &format!("#2677 latent binary row fifth, event={event}"),
            &exact,
            &difference,
        );
    }
}

/// The 24-row learned-scale stress family of the #2714 completion gate: exact-event
/// and right-censored rows, a loaded/unloaded split, four time columns and three
/// mean columns.
fn stress_survival_family(n: usize) -> LatentSurvivalFamily {
    LatentSurvivalFamily {
        event_target: Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1u8 } else { 0u8 })),
        weights: Array1::from_iter((0..n).map(|i| 0.55 + 0.03 * ((i % 7) as f64))),
        latent_sd_fixed: None,
        hazard_loading: HazardLoading::LoadedVsUnloaded,
        unloaded_mass_entry: Array1::from_iter((0..n).map(|i| 0.015 + 0.0015 * ((i % 11) as f64))),
        unloaded_mass_exit: Array1::from_iter((0..n).map(|i| 0.06 + 0.002 * ((i % 13) as f64))),
        unloaded_hazard_exit: Array1::from_iter((0..n).map(|i| {
            if i % 4 == 0 {
                0.018 + 0.001 * ((i % 5) as f64)
            } else {
                0.0
            }
        })),
        x_time_entry: Array2::from_shape_fn((n, 4), |(i, j)| {
            0.2 + 0.03 * ((i + 2 * j) % 9) as f64 - if j == 1 { 0.12 } else { 0.0 }
        }),
        x_time_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
            0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
        }),
        x_time_derivative_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
            0.45 + 0.015 * ((i + 3 * j) % 8) as f64
        }),
        x_time_right: Array2::from_shape_fn((n, 4), |(i, j)| {
            0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
        }),
        time_offset_right: Array1::zeros(n),
        unloaded_mass_right: Array1::zeros(n),
        x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_shape_fn((n, 3), |(i, j)| {
            0.1 + 0.04 * ((3 * i + j) % 7) as f64 - if j == 0 { 0.18 } else { 0.0 }
        }))),
        time_linear_constraints: None,
        quadctx: Arc::new(QuadratureContext::new()),
        baseline_theta_rows: None,
        jeffreys_armed: true,
    }
}

fn survival_states(family: &LatentSurvivalFamily, joint_beta: &Array1<f64>) -> Vec<ParameterBlockState> {
    let slices = family.joint_slices();
    let n = family.event_target.len();
    let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
    let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    eta_time.slice_mut(s![0..n]).assign(&family.x_time_entry.dot(&beta_time));
    eta_time.slice_mut(s![n..2 * n]).assign(&family.x_time_exit.dot(&beta_time));
    eta_time
        .slice_mut(s![2 * n..3 * n])
        .assign(&family.x_time_derivative_exit.dot(&beta_time));
    let mut states = vec![
        ParameterBlockState {
            beta: beta_time,
            eta: eta_time,
        },
        ParameterBlockState {
            beta: beta_mean.clone(),
            eta: family.x_mean.dot(&beta_mean),
        },
    ];
    if let Some(log_sigma) = slices.log_sigma {
        let beta_log_sigma = array![joint_beta[log_sigma.start]];
        states.push(ParameterBlockState {
            beta: beta_log_sigma.clone(),
            eta: beta_log_sigma,
        });
    }
    states
}

#[test]
fn latent_survival_third_information_derivative_matches_difference_of_second_2677() {
    let family = stress_survival_family(24);
    assert!(
        family.jeffreys_third_information_derivative().is_some(),
        "#2677: the armed latent survival family must declare its third information derivative"
    );
    let beta = array![0.18, 0.11, 0.07, 0.13, -0.09, 0.05, 0.12, 0.42_f64.ln()];
    let direction_u = array![0.21, -0.13, 0.08, 0.05, -0.17, 0.09, 0.04, -0.06];
    let direction_v = array![-0.07, 0.12, 0.15, -0.09, 0.06, -0.11, 0.08, 0.05];
    let states = survival_states(&family, &beta);
    let exact = family
        .exact_newton_joint_hessian_third_directional_derivative_all_axes_dense(
            &states,
            &direction_u,
            &direction_v,
        )
        .expect("latent survival third directional derivative");
    assert_eq!(exact.len(), beta.len());
    for (axis, exact_axis) in exact.iter().enumerate() {
        let difference = five_point_difference(
            |t| {
                let mut moved = beta.clone();
                moved[axis] += t;
                family
                    .exact_newton_joint_hessian_second_directional_derivative_dense(
                        &survival_states(&family, &moved),
                        &direction_u,
                        &direction_v,
                    )
                    .expect("latent survival second directional derivative at a moved beta")
            },
            DIFFERENCE_STEP,
        );
        assert_matches_difference(
            &format!("#2677 latent survival D3H[u, v, e_{axis}]"),
            exact_axis,
            &difference,
        );
    }
}

fn binary_family() -> LatentBinaryFamily {
    LatentBinaryFamily {
        event_target: array![1u8, 0u8, 1u8],
        weights: array![1.0, 0.7, 0.9],
        latent_sd: 0.35,
        hazard_loading: HazardLoading::LoadedVsUnloaded,
        unloaded_mass_entry: array![0.02, 0.03, 0.025],
        unloaded_mass_exit: array![0.05, 0.08, 0.06],
        x_time_entry: array![[1.0, -0.2], [0.4, 0.7], [0.6, 0.3]],
        x_time_exit: array![[1.3, 0.1], [0.9, 1.0], [1.1, 0.5]],
        x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, -0.3], [0.2, 0.9], [0.5, 0.4]])),
        time_linear_constraints: None,
        quadctx: Arc::new(QuadratureContext::new()),
        baseline_theta_rows: None,
        jeffreys_armed: true,
    }
}

fn binary_states(family: &LatentBinaryFamily, joint_beta: &Array1<f64>) -> Vec<ParameterBlockState> {
    let slices = family.joint_slices();
    let n = family.event_target.len();
    let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
    let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    eta_time.slice_mut(s![0..n]).assign(&family.x_time_entry.dot(&beta_time));
    eta_time.slice_mut(s![n..2 * n]).assign(&family.x_time_exit.dot(&beta_time));
    vec![
        ParameterBlockState {
            beta: beta_time,
            eta: eta_time,
        },
        ParameterBlockState {
            beta: beta_mean.clone(),
            eta: family.x_mean.dot(&beta_mean),
        },
    ]
}

#[test]
fn latent_binary_third_information_derivative_matches_difference_of_second_2677() {
    let family = binary_family();
    assert!(
        family.jeffreys_third_information_derivative().is_some(),
        "#2677: the armed latent binary family must declare its third information derivative"
    );
    // Every row's exit mass must exceed its entry mass: q_exit − q_entry is
    // (0.255, 0.375, 0.35) at this time block.
    let beta = array![0.6, 0.25, -0.2, 0.15];
    let direction_u = array![0.21, -0.13, 0.08, 0.05];
    let direction_v = array![-0.07, 0.12, 0.15, -0.09];
    let states = binary_states(&family, &beta);
    let exact = family
        .exact_newton_joint_hessian_third_directional_derivative_all_axes_dense(
            &states,
            &direction_u,
            &direction_v,
        )
        .expect("latent binary third directional derivative");
    assert_eq!(exact.len(), beta.len());
    for (axis, exact_axis) in exact.iter().enumerate() {
        let difference = five_point_difference(
            |t| {
                let mut moved = beta.clone();
                moved[axis] += t;
                family
                    .exact_newton_joint_hessian_second_directional_derivative_dense(
                        &binary_states(&family, &moved),
                        &direction_u,
                        &direction_v,
                    )
                    .expect("latent binary second directional derivative at a moved beta")
            },
            DIFFERENCE_STEP,
        );
        assert_matches_difference(
            &format!("#2677 latent binary D3H[u, v, e_{axis}]"),
            exact_axis,
            &difference,
        );
    }
}
