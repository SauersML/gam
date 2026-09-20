#![cfg(test)]
//! #2992: the latent-survival family forms its Jeffreys axis derivatives in the drift basis.
//!
//! The outer Jeffreys drift reads `{D[e_a]}` only as rows `vec(sym(Uᵀ D[e_a] U))`. The family
//! forms them from each row's primary lifts in one row pass. They must be the dense all-axes
//! derivatives rotated by `jeffreys_rotated_axis_rows`, for the first, second and third
//! information derivatives, at a full-width basis and a narrower non-orthogonal one, with σ
//! estimated and fixed.
//!
//! Both routes sum the same per-row terms `(X_i)_{γa} (Y_iᵀ F_{i,γ} Y_i)_{st}` in different
//! orders, so they agree to the rounding of those sums: `2γ_N` of the sum of the terms'
//! magnitudes, with `N` the number of rounded operations along the longer route. The magnitude
//! is formed here from the same kernels with every factor replaced by its absolute value.

use super::*;
use gam_math::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, array, s};

fn stress_survival_family(n: usize, estimate_sigma: bool) -> LatentSurvivalFamily {
    LatentSurvivalFamily {
        event_target: Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1u8 } else { 0u8 })),
        weights: Array1::from_iter((0..n).map(|i| 0.55 + 0.03 * ((i % 7) as f64))),
        latent_sd_fixed: if estimate_sigma { None } else { Some(0.42) },
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

/// The sum over rows and primaries of `|w_i| |(X_i)_{γa}| (|Y_i|ᵀ |F_{i,γ}| |Y_i|)_{st}`, with
/// `|Y_i| = |X_i| |U|`: a bound on the magnitude of every term either route sums.
fn absolute_axis_rows(
    family: &LatentSurvivalFamily,
    states: &[ParameterBlockState],
    basis: &Array2<f64>,
    lift: &dyn Fn(usize, &LatentSurvivalRow, LatentSurvivalPrimaryPoint, &Array1<f64>) -> Array2<f64>,
) -> Array2<f64> {
    let (q_entry, q_exit, qdot_exit, mu) = family.split_time_eta(states).expect("time eta");
    let q_right = family.time_q_right(states).expect("right time eta");
    let sigma = family.latent_sd(states).expect("latent sd");
    let slices = family.joint_slices();
    let (p, r) = basis.dim();
    let absolute_basis = basis.mapv(f64::abs);
    let mut magnitude = Array2::<f64>::zeros((p, r * r));
    for row_idx in 0..family.event_target.len() {
        let weight = family.weights[row_idx].abs();
        let jacobian = family.row_primary_jacobian(row_idx, &slices).expect("primary jacobian");
        let absolute_jacobian = jacobian.mapv(f64::abs);
        let projected = absolute_jacobian.dot(&absolute_basis);
        let row = family
            .build_row_at(row_idx, q_entry[row_idx], q_exit[row_idx], qdot_exit[row_idx], q_right[row_idx])
            .expect("latent survival row");
        let point = LatentSurvivalPrimaryPoint {
            q_entry: q_entry[row_idx],
            q_exit: q_exit[row_idx],
            qdot_exit: qdot_exit[row_idx],
            q_right: q_right[row_idx],
            mu: mu[row_idx],
            sigma,
        };
        for gamma in 0..LATENT_SURVIVAL_PRIMARY_DIM {
            if jacobian.row(gamma).iter().all(|&value| value == 0.0) {
                continue;
            }
            let mut seed = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
            seed[gamma] = 1.0;
            let kernel = lift(row_idx, &row, point, &seed).mapv(f64::abs) * weight;
            let reduced = projected.t().dot(&kernel).dot(&projected);
            for axis in 0..p {
                let loading = absolute_jacobian[[gamma, axis]];
                for s in 0..r {
                    for t in 0..r {
                        magnitude[[axis, s * r + t]] += loading * reduced[[s, t]];
                    }
                }
            }
        }
    }
    magnitude
}

/// Every entry of `rotated` equals `reference` to twice the rounding band of their shared
/// terms, and the rows are large enough against that band for agreement to say something.
fn assert_rotated_rows_match(
    label: &str,
    rotated: &Array2<f64>,
    reference: &Array2<f64>,
    magnitude: &Array2<f64>,
    operations: usize,
) {
    assert_eq!(rotated.dim(), reference.dim(), "{label}: one r x r row per axis");
    let growth = accumulation_growth(operations);
    let mut resolved = 0.0_f64;
    for ((axis, column), &got) in rotated.indexed_iter() {
        let want = reference[[axis, column]];
        let band = 2.0 * growth * magnitude[[axis, column]];
        assert!(
            (got - want).abs() <= band,
            "{label} axis {axis} entry {column}: rotated {got:+.17e}, dense {want:+.17e}, \
             band {band:.3e}"
        );
        if band > 0.0 {
            resolved = resolved.max(want.abs() / band);
        }
    }
    assert!(
        resolved > 1.0e6,
        "{label}: the largest row entry is only {resolved:.3e} bands, so agreement says nothing"
    );
}

fn bases(p: usize) -> [Array2<f64>; 2] {
    let full = Array2::from_shape_fn((p, p), |(i, j)| {
        (0.4 * ((i + 2 * j) as f64)).sin() + if i == j { 1.0 } else { 0.0 }
    });
    let narrow = Array2::from_shape_fn((p, 3), |(i, j)| (0.3 * (((i + 1) * (j + 2)) as f64)).cos());
    [full, narrow]
}

fn beta_and_directions(estimate_sigma: bool) -> (Array1<f64>, [Array1<f64>; 3]) {
    let mut beta = vec![0.18, 0.11, 0.07, 0.13, -0.09, 0.05, 0.12];
    let mut first = vec![0.21, -0.13, 0.08, 0.05, -0.17, 0.09, 0.04];
    let mut second = vec![-0.07, 0.12, 0.15, -0.09, 0.06, -0.11, 0.08];
    let mut third = vec![0.03, 0.09, -0.14, 0.11, 0.02, 0.16, -0.05];
    if estimate_sigma {
        beta.push(0.42_f64.ln());
        first.push(-0.06);
        second.push(0.05);
        third.push(0.07);
    }
    (Array1::from(beta), [Array1::from(first), Array1::from(second), Array1::from(third)])
}

/// Rounded operations along the longer route for one entry: every row's primary congruences
/// and the dense route's rotation.
fn operation_count(family: &LatentSurvivalFamily, p: usize) -> usize {
    family.event_target.len() * LATENT_SURVIVAL_PRIMARY_DIM.pow(3) + 2 * p * p
}

#[test]
fn latent_survival_rotated_first_information_rows_are_the_dense_axes_rotated_2992() {
    for estimate_sigma in [true, false] {
        let family = stress_survival_family(24, estimate_sigma);
        assert!(
            family.jeffreys_rotated_first_derivative().is_some(),
            "#2992: the latent survival family forms its rotated first information rows"
        );
        let (beta, _) = beta_and_directions(estimate_sigma);
        let states = survival_states(&family, &beta);
        let p = beta.len();
        let dense = family
            .exact_newton_joint_hessian_directional_derivative_all_axes_dense(&states)
            .expect("dense first information axes");
        let magnitude_lift = |_: usize, row: &LatentSurvivalRow, point, seed: &Array1<f64>| {
            latent_survival_row_primary_third_contracted(&family.quadctx, row, point, seed, estimate_sigma)
                .expect("third contracted")
        };
        for basis in bases(p) {
            let reference = gam_model_api::jeffreys_rotated_axis_rows(&dense, basis.view())
                .expect("rotated dense axes");
            let rotated = family
                .first_directional_rotated_axis_rows(&states, basis.view())
                .expect("rotated first rows");
            let magnitude = absolute_axis_rows(&family, &states, &basis, &magnitude_lift);
            assert_rotated_rows_match(
                &format!("#2992 rotated dH (sigma estimated {estimate_sigma}, width {})", basis.ncols()),
                &rotated,
                &reference,
                &magnitude,
                operation_count(&family, p),
            );
        }
    }
}

#[test]
fn latent_survival_rotated_second_information_rows_are_each_directions_dense_axes_rotated_2992() {
    for estimate_sigma in [true, false] {
        let family = stress_survival_family(24, estimate_sigma);
        let (beta, directions) = beta_and_directions(estimate_sigma);
        let states = survival_states(&family, &beta);
        let p = beta.len();
        for basis in bases(p) {
            let batch = family
                .second_directional_rotated_axis_rows_each(&states, &directions, basis.view())
                .expect("rotated second rows for the batch");
            assert_eq!(batch.len(), directions.len(), "one set of rows per direction");
            let slices = family.joint_slices();
            for (index, direction) in directions.iter().enumerate() {
                // The dense axes from the same per-primary lifts, closed by the production
                // all-axes reduction: the terms the rotated rows sum, in the dense route's order.
                let fourth_lift = |row_idx: usize, row: &LatentSurvivalRow, point, seed: &Array1<f64>| {
                    let primary = family.row_primary_direction_from_flat(row_idx, &slices, direction);
                    latent_survival_row_primary_fourth_contracted(&family.quadctx, row, point, &primary, seed, estimate_sigma)
                };
                let dense = family
                    .joint_hessian_axes_from_primary_lifts(&states, "contracted fourth", |lift| {
                        fourth_lift(lift.row_idx, lift.row, lift.point, lift.primary).map_err(String::from)
                    })
                    .expect("dense second information axes");
                let reference = gam_model_api::jeffreys_rotated_axis_rows(&dense, basis.view())
                    .expect("rotated dense axes");
                let magnitude = absolute_axis_rows(&family, &states, &basis, &|row_idx, row, point, seed| {
                    fourth_lift(row_idx, row, point, seed).expect("fourth contracted")
                });
                let direct: Vec<Array2<f64>> = (0..p)
                    .map(|axis| {
                        let mut unit = Array1::<f64>::zeros(p);
                        unit[axis] = 1.0;
                        family
                            .exact_newton_joint_hessian_second_directional_derivative_dense(&states, direction, &unit)
                            .expect("dense second information axis")
                    })
                    .collect();
                let direct = gam_model_api::jeffreys_rotated_axis_rows(&direct, basis.view()).expect("rotated direct axes");
                // The trait default's route, one row pass per axis with the axis as the kernel's
                // second seed: the per-direction oracle the Jeffreys drift read before (#2992).
                assert_rotated_rows_match(
                    &format!(
                        "#2992 rotated d2H direction {index} against the per-axis default (sigma estimated \
                         {estimate_sigma}, width {})",
                        basis.ncols()
                    ),
                    &batch[index],
                    &direct,
                    &magnitude,
                    operation_count(&family, p),
                );
                assert_rotated_rows_match(
                    &format!(
                        "#2992 rotated d2H direction {index} (sigma estimated {estimate_sigma}, width {})",
                        basis.ncols()
                    ),
                    &batch[index],
                    &reference,
                    &magnitude,
                    operation_count(&family, p),
                );
            }
        }
    }
}

#[test]
fn latent_survival_rotated_third_information_rows_are_the_dense_axes_rotated_2992() {
    for estimate_sigma in [true, false] {
        let family = stress_survival_family(24, estimate_sigma);
        let (beta, [direction_u, direction_v, _]) = beta_and_directions(estimate_sigma);
        let states = survival_states(&family, &beta);
        let p = beta.len();
        let dense = family
            .exact_newton_joint_hessian_third_directional_derivative_all_axes_dense(&states, &direction_u, &direction_v)
            .expect("dense third information axes");
        let slices = family.joint_slices();
        for basis in bases(p) {
            let reference = gam_model_api::jeffreys_rotated_axis_rows(&dense, basis.view())
                .expect("rotated dense axes");
            let rotated = family
                .third_directional_rotated_axis_rows(&states, &direction_u, &direction_v, basis.view())
                .expect("rotated third rows");
            let magnitude = absolute_axis_rows(&family, &states, &basis, &|row_idx, row, point, seed| {
                let u = family.row_primary_direction_from_flat(row_idx, &slices, &direction_u);
                let v = family.row_primary_direction_from_flat(row_idx, &slices, &direction_v);
                latent_survival_row_primary_fifth_contracted(
                    &family.quadctx,
                    row,
                    point,
                    &u,
                    &v,
                    seed,
                    estimate_sigma,
                )
                .expect("fifth contracted")
            });
            assert_rotated_rows_match(
                &format!("#2992 rotated d3H (sigma estimated {estimate_sigma}, width {})", basis.ncols()),
                &rotated,
                &reference,
                &magnitude,
                operation_count(&family, p),
            );
        }
    }
}
