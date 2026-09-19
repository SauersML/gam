//! Survival location-scale derivative identities: the all-axes and per-axis directional
//! derivatives of the observed information against differences of lower orders and against
//! the dense axes contracted (#2668, #2677), and the closed-form ratio, log-survival and log-pdf
//! derivative stacks against independent finite-difference witnesses.
#![cfg(test)]

use super::*;

/// #2677: the all-axes third directional derivative of the observed
/// information is the β-derivative of its second directional derivative. For
/// every residual distribution with closed-form fifth stacks, a five-point
/// difference of `I''[u, v]` along each coefficient axis reproduces
/// `{I'''[u, v, e_a]}` built from the row program's fifth-order contraction,
/// and the family declares that channel available. LogLog and Cauchit read
/// their fourth-order stacks from the generic pdf-jet dispatch and their fifth
/// from the Bernoulli tail kernels (#2903).
#[test]
fn survival_ls_third_directional_all_axes_matches_difference_of_second_2677() {
    use crate::custom_family::CustomFamily;
    use crate::row_kernel::{RowSet, row_kernel_third_directional_derivative_all_axes};

    let beta = [0.3, -0.4, 0.2];
    let u = array![0.7, -0.5, 0.9];
    let v = array![-0.4, 1.1, 0.6];
    for distribution in [
        residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        residual_distribution_inverse_link(ResidualDistribution::Gumbel),
        residual_distribution_inverse_link(ResidualDistribution::Logistic),
        InverseLink::Standard(StandardLink::LogLog),
        InverseLink::Standard(StandardLink::Cauchit),
    ] {
        let family = survival_exact_newton_test_familywith_inverse_link(distribution.clone());
        assert!(
            family.jeffreys_third_information_derivative().is_some(),
            "{distribution:?}: a closed-form link must declare the third information derivative"
        );
        let states = survival_exact_newton_test_states(&family, beta[0], beta[1], beta[2]);
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let axes = row_kernel_third_directional_derivative_all_axes(
            &kernel,
            &RowSet::All,
            u.as_slice().expect("contiguous u"),
            v.as_slice().expect("contiguous v"),
        )
        .expect("third directional derivative");
        assert_eq!(axes.len(), beta.len(), "{distribution:?}: one matrix per axis");
        let second_at = |axis: usize, t: f64| {
            let mut moved = beta;
            moved[axis] += t;
            let moved_states =
                survival_exact_newton_test_states(&family, moved[0], moved[1], moved[2]);
            family
                .exact_newton_joint_hessian_second_directional_derivative_rescaled(
                    &moved_states,
                    &u,
                    &v,
                    0.0,
                )
                .expect("second directional derivative")
                .expect("second directional derivative present")
        };
        let h = 1.0e-3;
        let mut largest = 0.0_f64;
        for axis in 0..beta.len() {
            let difference = (-second_at(axis, 2.0 * h) + 8.0 * second_at(axis, h)
                - 8.0 * second_at(axis, -h)
                + second_at(axis, -2.0 * h))
                / (12.0 * h);
            for ((a, b), &want) in difference.indexed_iter() {
                let got = axes[axis][[a, b]];
                largest = largest.max(got.abs());
                assert!(
                    (got - want).abs() <= 1.0e-6 * (1.0 + want.abs().max(got.abs())),
                    "{distribution:?} axis {axis} I'''[{a}][{b}]: generated {got:+.15e}, \
                     difference {want:+.15e}"
                );
            }
        }
        assert!(
            largest > 1.0e-3,
            "{distribution:?}: the third information derivative is too small ({largest:.3e}) \
             for the agreement to say anything"
        );
    }
}

/// #2668/#2106: the fused Jeffreys contracted trace Hessian is exactly the pairwise
/// contraction it replaces. Every entry `[∇²_β tr(W · I)][c][d]` must equal
/// `tr(W · I''[e_c, e_d])` assembled from the family's own second directional
/// derivative, for each closed-form residual distribution. The weight is signed
/// and not symmetric, so the linear-in-`W` claim is exercised on the case a
/// factorizing implementation would get wrong, and the family advertises the
/// hook on the non-wiggle row kernel.
#[test]
fn survival_ls_contracted_trace_hessian_matches_pairwise_second_directional_2668() {
    use crate::custom_family::CustomFamily;
    use crate::row_kernel::{RowSet, row_kernel_contracted_trace_hessian};

    let weight = array![[0.9, -0.3, 0.45], [0.2, -1.1, 0.35], [-0.6, 0.25, 0.7]];
    let p = weight.nrows();
    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(distribution),
        );
        assert!(
            family.joint_jeffreys_information_contracted_trace_hessian_available(),
            "{distribution:?}: the non-wiggle row kernel must advertise the fused contraction"
        );
        let states = survival_exact_newton_test_states(&family, 0.3, -0.4, 0.2);
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let fused = row_kernel_contracted_trace_hessian(&kernel, &RowSet::All, &weight)
            .expect("contracted trace Hessian");
        assert_eq!(fused.dim(), (p, p), "{distribution:?}: one p×p matrix");
        let mut largest = 0.0_f64;
        for c in 0..p {
            for d in 0..p {
                let mut e_c = Array1::<f64>::zeros(p);
                e_c[c] = 1.0;
                let mut e_d = Array1::<f64>::zeros(p);
                e_d[d] = 1.0;
                let second = family
                    .exact_newton_joint_hessian_second_directional_derivative_rescaled(
                        &states, &e_c, &e_d, 0.0,
                    )
                    .expect("second directional derivative")
                    .expect("second directional derivative present");
                // tr(W · H) = Σ_ab W[a][b] · H[b][a].
                let want = (&weight * &second.t()).sum();
                let got = fused[[c, d]];
                largest = largest.max(want.abs());
                assert!(
                    (got - want).abs() <= 1.0e-10 * (1.0 + want.abs().max(got.abs())),
                    "{distribution:?} [{c}][{d}]: fused {got:+.15e}, pairwise {want:+.15e}"
                );
            }
        }
        assert!(
            largest > 1.0e-3,
            "{distribution:?}: the contracted second derivative is too small ({largest:.3e}) \
             for the agreement to say anything"
        );
    }
}

/// #2668/#2106: the batched all-axes second directional derivative that the Jeffreys
/// drift consumes is the per-axis hook, not an approximation of it. The dispatcher
/// builds the geometry once and runs the same row fold per axis, so every matrix
/// must equal `I''[u, e_a]` from the per-axis hook exactly, for each closed-form
/// residual distribution.
#[test]
fn survival_ls_all_axes_second_directional_is_the_per_axis_hook_2668() {
    use crate::row_kernel::{RowSet, row_kernel_second_directional_derivative_all_axes};

    let u = array![0.7, -0.5, 0.9];
    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(distribution),
        );
        let states = survival_exact_newton_test_states(&family, 0.3, -0.4, 0.2);
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let batched = row_kernel_second_directional_derivative_all_axes(
            &kernel,
            &RowSet::All,
            u.as_slice().expect("contiguous u"),
        )
        .expect("all-axes second directional derivative");
        let p = u.len();
        assert_eq!(batched.len(), p, "{distribution:?}: one matrix per axis");
        let mut largest = 0.0_f64;
        for (axis, matrix) in batched.iter().enumerate() {
            let mut e_a = Array1::<f64>::zeros(p);
            e_a[axis] = 1.0;
            let per_axis = family
                .exact_newton_joint_hessian_second_directional_derivative_rescaled(
                    &states, &u, &e_a, 0.0,
                )
                .expect("per-axis second directional derivative")
                .expect("per-axis second directional derivative present");
            largest = largest.max(per_axis.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())));
            assert_eq!(
                matrix, &per_axis,
                "{distribution:?} axis {axis}: the batched matrix must be the per-axis hook's"
            );
        }
        assert!(
            largest > 1.0e-3,
            "{distribution:?}: I''[u, e_a] is too small ({largest:.3e}) for equality to say anything"
        );
    }
}

/// #2668: `SurvivalLsRowKernel::second_directional_derivative_all_axes_dense_override`
/// builds each row's plan, channel rows and `J·u` once and reduces in chunk order. That is
/// only a valid optimisation if every matrix is the generic per-axis fold's, bit for bit.
/// The 3-row fixture above never leaves one chunk, so this pin uses `n = 300` rows (the
/// `ARROW_ROW_CHUNK = 256` reduction spans two tiles), multi-column threshold and log-σ
/// designs, mixed event and censored rows and non-unit weights. It also asserts that the
/// full-data dispatcher actually takes the override.
#[test]
fn survival_ls_all_axes_second_directional_override_is_the_per_axis_fold_across_chunks_2668() {
    use crate::row_kernel::{
        RowKernel, RowSet, row_kernel_second_directional_derivative,
        row_kernel_second_directional_derivative_all_axes,
    };

    let n = 300usize;
    let p_thr = 3usize;
    let p_ls = 3usize;
    let x_time_entry = Array2::from_elem((n, 1), 0.7);
    let x_time_exit =
        Array2::from_shape_fn((n, 1), |(r, _)| 1.2 + 0.4 * ((r as f64) * 0.37).sin());
    let x_time_deriv = Array2::from_elem((n, 1), 1.0);
    let x_threshold = Array2::from_shape_fn((n, p_thr), |(r, j)| {
        0.3 + 0.5 * ((r as f64) * 0.11 + j as f64).cos() - 0.02 * (j as f64)
    });
    let x_log_sigma = Array2::from_shape_fn((n, p_ls), |(r, j)| {
        0.1 + 0.4 * ((r as f64) * 0.07 - 0.5 * (j as f64)).sin()
    });
    let beta_t = array![0.3];
    let beta_thr = array![-0.4, 0.25, 0.1];
    let beta_ls = array![0.2, -0.15, 0.05];
    let u = array![0.7, -0.5, 0.9, 0.3, -0.2, 0.6, -0.4];
    let u_slice = u.as_slice().expect("contiguous u");
    let p = u.len();

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let mut family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(distribution),
        );
        family.n = n;
        family.entry_active = Arc::from(vec![true; n]);
        family.y = Array1::from_iter((0..n).map(|r| if r % 3 == 0 { 0.0 } else { 1.0 }));
        family.w = Array1::from_iter((0..n).map(|r| 0.6 + 0.1 * ((r % 7) as f64)));
        family.x_time_entry = Arc::new(x_time_entry.clone());
        family.x_time_exit = Arc::new(x_time_exit.clone());
        family.x_time_deriv = Arc::new(x_time_deriv.clone());
        family.x_threshold =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_threshold.clone()));
        family.x_log_sigma =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_log_sigma.clone()));
        // Stacked time eta layout `[entry; exit; deriv]`, as `survival_exact_newton_test_states`.
        let mut eta_time = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta_time[i] = x_time_entry[[i, 0]] * beta_t[0];
            eta_time[n + i] = x_time_exit[[i, 0]] * beta_t[0];
            eta_time[2 * n + i] = x_time_deriv[[i, 0]] * beta_t[0];
        }
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_thr.clone(),
                eta: x_threshold.dot(&beta_thr),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: x_log_sigma.dot(&beta_ls),
            },
        ];
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        assert!(
            matches!(
                kernel.second_directional_derivative_all_axes_dense_override(&RowSet::All, u_slice),
                Some(Ok(_))
            ),
            "{distribution:?}: the full-data dispatcher must take the build-once override"
        );
        let batched =
            row_kernel_second_directional_derivative_all_axes(&kernel, &RowSet::All, u_slice)
                .expect("all-axes second directional derivative");
        assert_eq!(batched.len(), p, "{distribution:?}: one matrix per axis");
        let mut largest = 0.0_f64;
        for (axis, matrix) in batched.iter().enumerate() {
            let mut e_a = vec![0.0_f64; p];
            e_a[axis] = 1.0;
            let per_axis =
                row_kernel_second_directional_derivative(&kernel, &RowSet::All, u_slice, &e_a)
                    .expect("per-axis second directional fold");
            largest = largest.max(per_axis.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())));
            assert_eq!(
                matrix, &per_axis,
                "{distribution:?} axis {axis}: the override must be the per-axis fold bit for bit"
            );
        }
        assert!(
            largest > 1.0e-3,
            "{distribution:?}: I''[u, e_a] is too small ({largest:.3e}) for equality to say anything"
        );
    }
}

/// #2668: the survival location-scale contraction pass `⟨H²dot[u, e_a], K_b⟩`, which the
/// Jeffreys drift consumes in place of the rotated axis rows, is the dense all-axes matrices
/// contracted with the same kernels. The pass forms no axis matrix and sums in a different
/// order, so agreement is to roundoff. It is checked on the `n = 300` two-tile fixture with
/// multi-column blocks, two directions and non-diagonal symmetric kernels.
#[test]
fn survival_ls_second_directional_axis_contractions_are_the_dense_axes_contracted_2668() {
    use crate::row_kernel::{RowSet, row_kernel_second_directional_derivative_all_axes};

    let n = 300usize;
    let p_thr = 3usize;
    let p_ls = 3usize;
    let x_time_entry = Array2::from_elem((n, 1), 0.7);
    let x_time_exit =
        Array2::from_shape_fn((n, 1), |(r, _)| 1.2 + 0.4 * ((r as f64) * 0.37).sin());
    let x_time_deriv = Array2::from_elem((n, 1), 1.0);
    let x_threshold = Array2::from_shape_fn((n, p_thr), |(r, j)| {
        0.3 + 0.5 * ((r as f64) * 0.11 + j as f64).cos() - 0.02 * (j as f64)
    });
    let x_log_sigma = Array2::from_shape_fn((n, p_ls), |(r, j)| {
        0.1 + 0.4 * ((r as f64) * 0.07 - 0.5 * (j as f64)).sin()
    });
    let beta_t = array![0.3];
    let beta_thr = array![-0.4, 0.25, 0.1];
    let beta_ls = array![0.2, -0.15, 0.05];
    let directions = vec![
        array![0.7, -0.5, 0.9, 0.3, -0.2, 0.6, -0.4],
        array![-0.3, 0.8, 0.1, -0.6, 0.4, 0.2, 0.5],
    ];
    let p = directions[0].len();
    let kernels: Vec<Array2<f64>> = (0..p)
        .map(|b| {
            Array2::from_shape_fn((p, p), |(i, j)| {
                (0.3 * ((b + 1) as f64) * ((i + j) as f64)).cos() + if i == j { 1.0 } else { 0.0 }
            })
        })
        .collect();

    for distribution in [ResidualDistribution::Gaussian, ResidualDistribution::Logistic] {
        let mut family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(distribution),
        );
        family.n = n;
        family.entry_active = Arc::from(vec![true; n]);
        family.y = Array1::from_iter((0..n).map(|r| if r % 3 == 0 { 0.0 } else { 1.0 }));
        family.w = Array1::from_iter((0..n).map(|r| 0.6 + 0.1 * ((r % 7) as f64)));
        family.x_time_entry = Arc::new(x_time_entry.clone());
        family.x_time_exit = Arc::new(x_time_exit.clone());
        family.x_time_deriv = Arc::new(x_time_deriv.clone());
        family.x_threshold =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_threshold.clone()));
        family.x_log_sigma =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_log_sigma.clone()));
        let mut eta_time = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta_time[i] = x_time_entry[[i, 0]] * beta_t[0];
            eta_time[n + i] = x_time_exit[[i, 0]] * beta_t[0];
            eta_time[2 * n + i] = x_time_deriv[[i, 0]] * beta_t[0];
        }
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_thr.clone(),
                eta: x_threshold.dot(&beta_thr),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: x_log_sigma.dot(&beta_ls),
            },
        ];
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let mut contracted: Vec<Option<Array2<f64>>> = vec![None; directions.len()];
        kernel
            .second_directional_axis_contractions_each(&directions, &kernels, &mut |index, matrix| {
                contracted[index] = Some(matrix);
                Ok(())
            })
            .expect("axis contraction pass");
        for (index, direction) in directions.iter().enumerate() {
            let axes = row_kernel_second_directional_derivative_all_axes(
                &kernel,
                &RowSet::All,
                direction.as_slice().expect("contiguous direction"),
            )
            .expect("dense all-axes second directional derivative");
            let got = contracted[index]
                .as_ref()
                .expect("the pass hands over every direction");
            assert_eq!(got.dim(), (p, p), "{distribution:?}: one p x p contraction matrix");
            let mut largest = 0.0_f64;
            for a in 0..p {
                for b in 0..p {
                    let dense: f64 = axes[a].iter().zip(kernels[b].iter()).map(|(x, y)| x * y).sum();
                    largest = largest.max(dense.abs());
                    assert!(
                        (got[[a, b]] - dense).abs() <= 1.0e-10 * (1.0 + dense.abs()),
                        "{distribution:?} direction {index} axis {a} kernel {b}: contraction {} \
                         vs dense {dense}",
                        got[[a, b]]
                    );
                }
            }
            assert!(
                largest > 1.0e-3,
                "{distribution:?} direction {index}: contractions too small ({largest:.3e}) for \
                 agreement to say anything"
            );
        }
    }
}

/// #2668: survival location-scale hands the Jeffreys term, its drift base and the gate motion
/// the rotated first information rows `vec(sym(Uᵀ I'[e_a] U))` from each row's third
/// contractions and projected channel rows, not from `p` dense axis matrices. They must be the
/// dense all-axes matrices rotated and symmetrized, to roundoff, on the `n = 300` two-tile
/// fixture with multi-column blocks, for a full-width basis and a narrower non-orthogonal one.
#[test]
fn survival_ls_rotated_first_directional_rows_are_the_dense_axes_rotated_2668() {
    use crate::row_kernel::{RowSet, row_kernel_directional_derivative_all_axes};

    let n = 300usize;
    let p_thr = 3usize;
    let p_ls = 3usize;
    let x_time_entry = Array2::from_elem((n, 1), 0.7);
    let x_time_exit =
        Array2::from_shape_fn((n, 1), |(r, _)| 1.2 + 0.4 * ((r as f64) * 0.37).sin());
    let x_time_deriv = Array2::from_elem((n, 1), 1.0);
    let x_threshold = Array2::from_shape_fn((n, p_thr), |(r, j)| {
        0.3 + 0.5 * ((r as f64) * 0.11 + j as f64).cos() - 0.02 * (j as f64)
    });
    let x_log_sigma = Array2::from_shape_fn((n, p_ls), |(r, j)| {
        0.1 + 0.4 * ((r as f64) * 0.07 - 0.5 * (j as f64)).sin()
    });
    let beta_t = array![0.3];
    let beta_thr = array![-0.4, 0.25, 0.1];
    let beta_ls = array![0.2, -0.15, 0.05];
    let p = 1 + p_thr + p_ls;
    let full = Array2::from_shape_fn((p, p), |(i, j)| {
        (0.4 * ((i + 2 * j) as f64)).sin() + if i == j { 1.0 } else { 0.0 }
    });
    let narrow =
        Array2::from_shape_fn((p, 3), |(i, j)| (0.3 * (((i + 1) * (j + 2)) as f64)).cos());

    for distribution in [ResidualDistribution::Gaussian, ResidualDistribution::Logistic] {
        let mut family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(distribution),
        );
        family.n = n;
        family.entry_active = Arc::from(vec![true; n]);
        family.y = Array1::from_iter((0..n).map(|r| if r % 3 == 0 { 0.0 } else { 1.0 }));
        family.w = Array1::from_iter((0..n).map(|r| 0.6 + 0.1 * ((r % 7) as f64)));
        family.x_time_entry = Arc::new(x_time_entry.clone());
        family.x_time_exit = Arc::new(x_time_exit.clone());
        family.x_time_deriv = Arc::new(x_time_deriv.clone());
        family.x_threshold =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_threshold.clone()));
        family.x_log_sigma =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_log_sigma.clone()));
        let mut eta_time = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta_time[i] = x_time_entry[[i, 0]] * beta_t[0];
            eta_time[n + i] = x_time_exit[[i, 0]] * beta_t[0];
            eta_time[2 * n + i] = x_time_deriv[[i, 0]] * beta_t[0];
        }
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_thr.clone(),
                eta: x_threshold.dot(&beta_thr),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: x_log_sigma.dot(&beta_ls),
            },
        ];
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let axes = row_kernel_directional_derivative_all_axes(&kernel, &RowSet::All)
            .expect("dense all-axes first directional derivative");
        assert_eq!(axes.len(), p, "{distribution:?}: one axis matrix per coefficient");
        for basis in [&full, &narrow] {
            let r = basis.ncols();
            let rotated = kernel
                .directional_derivative_rotated_all_axes(basis.view())
                .expect("rotated axis rows");
            assert_eq!(rotated.dim(), (p, r * r), "{distribution:?}: one r x r row per axis");
            let mut largest = 0.0_f64;
            for (a, axis) in axes.iter().enumerate() {
                let reduced = basis.t().dot(axis).dot(basis);
                for s in 0..r {
                    for t in 0..r {
                        let want = 0.5 * (reduced[[s, t]] + reduced[[t, s]]);
                        let got = rotated[[a, s * r + t]];
                        largest = largest.max(want.abs());
                        assert!(
                            (got - want).abs() <= 1.0e-10 * (1.0 + want.abs()),
                            "{distribution:?} width {r} axis {a} [{s}][{t}]: rotated {got:+.15e}, \
                             dense {want:+.15e}"
                        );
                    }
                }
            }
            assert!(
                largest > 1.0e-3,
                "{distribution:?} width {r}: rotated rows too small ({largest:.3e}) for agreement \
                 to say anything"
            );
        }
    }
}

#[test]
fn survival_ratio_derivatives_prefer_correct_signs() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.2, -0.5, 0.4, 0.6, 1.1];
    let h = 1e-6_f64;
    let tie_tol = 1e-12_f64;
    let nondeg_tol = 1e-12_f64;
    let mut saw_strict_dr = false;
    let mut saw_strict_ddr = false;

    for &dist in &dists {
        for &z in &zs {
            let r = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                f / s
            };
            let dr_plus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let ratio = f / s;
                (ratio * ratio) + fp / s
            };
            let dr_minus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let ratio = f / s;
                (ratio * ratio) - fp / s
            };
            let ddr_plus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let fpp = dist.pdfsecond_derivative(u);
                let ratio = f / s;
                let dr = (ratio * ratio) + fp / s;
                (2.0 * ratio * dr) + (fpp / s + fp * f / (s * s))
            };
            let ddr_minus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let fpp = dist.pdfsecond_derivative(u);
                let ratio = f / s;
                let dr = (ratio * ratio) - fp / s;
                (2.0 * ratio * dr) - (fpp / s + fp * f / (s * s))
            };

            let drfd = (r(z + h) - r(z - h)) / (2.0 * h);
            let ddrfd = (dr_plus(z + h) - dr_plus(z - h)) / (2.0 * h);
            let dr_plus_err = (dr_plus(z) - drfd).abs();
            let dr_minus_err = (dr_minus(z) - drfd).abs();
            let ddr_plus_err = (ddr_plus(z) - ddrfd).abs();
            let ddr_minus_err = (ddr_minus(z) - ddrfd).abs();
            let f = dist.pdf(z);
            let s = 1.0 - dist.cdf(z);
            let fp = dist.pdf_derivative(z);
            let fpp = dist.pdfsecond_derivative(z);
            let dr_signal = (fp / s).abs();
            let ddr_signal = (fpp / s + fp * f / (s * s)).abs();

            if dr_signal > nondeg_tol {
                saw_strict_dr = true;
                assert!(
                    dr_plus_err + tie_tol < dr_minus_err,
                    "dr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    dr_plus_err,
                    dr_minus_err,
                    dr_signal
                );
            } else {
                // At stationary points (fp≈0), plus/minus formulas coincide to first order.
                assert!(
                    (dr_plus_err - dr_minus_err).abs() <= tie_tol,
                    "dr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    dr_plus_err,
                    dr_minus_err,
                    dr_signal
                );
            }

            if ddr_signal > nondeg_tol {
                saw_strict_ddr = true;
                assert!(
                    ddr_plus_err + tie_tol < ddr_minus_err,
                    "ddr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    ddr_plus_err,
                    ddr_minus_err,
                    ddr_signal
                );
            } else {
                assert!(
                    (ddr_plus_err - ddr_minus_err).abs() <= tie_tol,
                    "ddr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    ddr_plus_err,
                    ddr_minus_err,
                    ddr_signal
                );
            }
        }
    }

    assert!(
        saw_strict_dr,
        "expected at least one non-degenerate dr check"
    );
    assert!(
        saw_strict_ddr,
        "expected at least one non-degenerate ddr check"
    );
}

#[test]
fn neglog_survival_stack_matches_closed_form_ratio_identities() {
    // The jet-composed `-ln S` stack must reproduce the classical quotient-rule
    // identities for `r = f/S` — `r' = r² + f'/S`, `r'' = 2rr' + f''/S + f'f/S²` —
    // and, at fourth order, a central difference of that closed-form `r''`.
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.4, -0.7, -0.1, 0.3, 0.9, 1.4];
    let closed_ddr = |dist: &ResidualDistribution, z: f64| -> f64 {
        let f = dist.pdf(z);
        let s = 1.0 - dist.cdf(z);
        let fp = dist.pdf_derivative(z);
        let fpp = dist.pdfsecond_derivative(z);
        let r = f / s;
        let dr = r * r + fp / s;
        2.0 * r * dr + (fpp / s + fp * f / (s * s))
    };

    for &dist in &dists {
        for &z in &zs {
            let f = dist.pdf(z);
            let s = 1.0 - dist.cdf(z);
            let fp = dist.pdf_derivative(z);
            let fpp = dist.pdfsecond_derivative(z);
            let fppp = dist.pdfthird_derivative(z);

            let (log_s, r, dr, ddr, dddr) =
                SurvivalLocationScaleFamily::neglog_survival_stack_from_pdf_jet(s, f, fp, fpp, fppp);

            let r_expected = f / s;
            let dr_expected = (r_expected * r_expected) + fp / s;
            let ddr_expected = closed_ddr(&dist, z);
            let h = 1e-4;
            let dddr_expected = (closed_ddr(&dist, z + h) - closed_ddr(&dist, z - h)) / (2.0 * h);

            assert!(
                (log_s - s.ln()).abs() <= 1e-14 * s.ln().abs().max(1.0),
                "log S mismatch for {dist:?} at z={z}: got {log_s}, expected {}",
                s.ln()
            );
            assert!(
                (r - r_expected).abs() <= 1e-14 * r_expected.abs().max(1.0),
                "r mismatch for {dist:?} at z={z}: got {r}, expected {r_expected}"
            );
            assert!(
                (dr - dr_expected).abs() <= 1e-12 * dr_expected.abs().max(1.0),
                "dr mismatch for {dist:?} at z={z}: got {dr}, expected {dr_expected}"
            );
            assert!(
                (ddr - ddr_expected).abs() <= 1e-10 * ddr_expected.abs().max(1.0),
                "ddr mismatch for {dist:?} at z={z}: got {ddr}, expected {ddr_expected}"
            );
            assert!(
                (dddr - dddr_expected).abs() <= 1e-6 * dddr_expected.abs().max(1.0),
                "dddr mismatch for {dist:?} at z={z}: got {dddr}, central difference {dddr_expected}"
            );
        }
    }
}

#[test]
fn residual_pdfthird_derivative_matchessecond_derivativefd() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.1, -0.4, 0.2, 0.9];
    let h = 1e-6_f64;

    for &dist in &dists {
        for &z in &zs {
            let fd =
                (dist.pdfsecond_derivative(z + h) - dist.pdfsecond_derivative(z - h)) / (2.0 * h);
            let analytic = dist.pdfthird_derivative(z);
            assert_eq!(
                analytic.signum(),
                fd.signum(),
                "pdf''' sign mismatch for {:?} at z={}: analytic={} fd={}",
                dist,
                z,
                analytic,
                fd
            );
            assert!(
                (analytic - fd).abs() < 5e-5,
                "pdf''' mismatch for {:?} at z={}: analytic={} fd={}",
                dist,
                z,
                analytic,
                fd
            );
        }
    }
}

/// #932: independent finite-difference witness of the residual-distribution
/// **fourth** PDF derivative `f''''(z)` for every residual distribution.
///
/// `pdfthird_derivative` was directly FD-guarded
/// (`residual_pdfthird_derivative_matchessecond_derivativefd`) but
/// `pdffourth_derivative` — the highest-order, most error-dense scalar tower
/// feeding the survival-LS outer-Hessian `m4` term — was only covered
/// transitively through the row-kernel oracle, where a sign slip can cancel
/// against another term. This pins it directly: a Richardson O(h⁴) central
/// difference of `pdfthird_derivative` (independent of the closed-form fourth)
/// must match `pdffourth_derivative`, and a planted sign flip must be rejected.
#[test]
fn residual_pdffourth_derivative_matches_independent_fd_witness() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.3_f64, -0.5, 0.3, 1.1];
    // Richardson-extrapolated central difference of f'''(z): cancels the O(h²)
    // error of the plain central stencil, giving an O(h⁴) witness independent of
    // the analytic fourth-derivative code path.
    let central = |dist: &ResidualDistribution, z: f64, h: f64| {
        (dist.pdfthird_derivative(z + h) - dist.pdfthird_derivative(z - h)) / (2.0 * h)
    };
    for &dist in &dists {
        for &z in &zs {
            let h = 1e-3_f64;
            let coarse = central(&dist, z, h);
            let fine = central(&dist, z, h * 0.5);
            let fd = (4.0 * fine - coarse) / 3.0;
            let analytic = dist.pdffourth_derivative(z);
            assert!(
                (analytic - fd).abs() <= 1e-4 * analytic.abs().max(1.0) + 1e-7,
                "pdf'''' mismatch for {dist:?} at z={z}: analytic={analytic} fd={fd}"
            );
            // Planted-corruption tripwire: a sign flip must leave the witness band.
            if analytic.abs() > 1e-6 {
                let corrupted = -analytic;
                assert!(
                    (corrupted - fd).abs() > 1e-4 * analytic.abs().max(1.0) + 1e-7,
                    "witness failed to reject a planted pdf'''' sign flip for {dist:?} at z={z}"
                );
            }
        }
    }
}

/// #932: independent finite-difference witness of the log-survival and
/// log-pdf scalar derivative stacks across all residual links.
///
/// The survival-LS row oracle (`SurvivalLsJointNllProgram`) seeds its tower
/// from `exact_survival_neglog_derivatives_fourth_rescaled` /
/// `exact_log_pdf_derivatives_rescaled`, so it tests the Faà-di-Bruno
/// composition but TRUSTS those scalar stacks as inputs. Outside the
/// identity/probit closed-form special cases they had no general independent
/// witness. This pins each stack's d1..d4 by differencing its OWN value
/// channel (the value is independently anchored by the closed-form tests):
/// a Richardson-extrapolated central stencil of `log S(eta)` / `log f(eta)`
/// must reproduce the analytic derivative channels for logit / probit / cloglog
/// over a range of eta, and a planted sign flip must be rejected.
#[test]
fn survival_log_survival_and_pdf_stacks_match_independent_fd_witness() {
    // LogLog and Cauchit reach the generic (jet-composed) arm of both stacks,
    // which the closed-form links never exercise; without them that arm had
    // no independent witness at all.
    let links = [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
        InverseLink::Standard(StandardLink::LogLog),
        InverseLink::Standard(StandardLink::Cauchit),
    ];
    let etas = [-0.8_f64, -0.2, 0.4, 1.0];

    // Richardson O(h⁴) central stencil of an arbitrary scalar f(eta) to the
    // requested derivative order (1..=4).
    fn stencil(order: usize) -> &'static [(i64, f64)] {
        match order {
            1 => &[(-1, -0.5), (1, 0.5)],
            2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
            3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
            4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
            _ => panic!("stencil supports derivative orders 1..=4, got {order}"),
        }
    }
    let central = |value: &dyn Fn(f64) -> f64, eta: f64, order: usize, h: f64| -> f64 {
        let one = |hh: f64| {
            stencil(order)
                .iter()
                .map(|&(off, c)| c * value(eta + (off as f64) * hh))
                .sum::<f64>()
                / hh.powi(order as i32)
        };
        (4.0 * one(h * 0.5) - one(h)) / 3.0
    };

    for link in &links {
        // log S(eta): value = slot 0; analytic derivatives are -r, -dr, -ddr, -dddr.
        let log_s_value = |eta: f64| {
            SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
                link, eta, 0.0,
            )
            .expect("log-survival stack")
            .0
        };
        // log f(eta): value = slot 0; analytic derivatives are d1..d4.
        let log_pdf_value = |eta: f64| {
            SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, eta, 0.0)
                .expect("log-pdf stack")
                .0
        };
        for &eta in &etas {
            let (_, r, dr, ddr, dddr) =
                SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
                    link, eta, 0.0,
                )
                .expect("log-survival stack");
            let log_s_analytic = [-r, -dr, -ddr, -dddr];
            let (_, p1, p2, p3, p4) =
                SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, eta, 0.0)
                    .expect("log-pdf stack");
            let log_pdf_analytic = [p1, p2, p3, p4];

            for (k, &analytic) in log_s_analytic.iter().enumerate() {
                let order = k + 1;
                let h = match order {
                    1 | 2 => 1e-3,
                    3 => 3e-3,
                    4 => 1e-2,
                    _ => unreachable!("stencil supports derivative orders 1..=4"),
                };
                let fd = central(&log_s_value, eta, order, h);
                assert!(
                    (analytic - fd).abs() <= 5e-4 * analytic.abs().max(1.0) + 1e-6,
                    "logS d{order} mismatch for {link:?} at eta={eta}: analytic={analytic} fd={fd}"
                );
                if analytic.abs() > 1e-5 {
                    assert!(
                        (-analytic - fd).abs() > 5e-4 * analytic.abs().max(1.0) + 1e-6,
                        "witness failed to reject logS d{order} sign flip for {link:?} at eta={eta}"
                    );
                }
            }
            for (k, &analytic) in log_pdf_analytic.iter().enumerate() {
                let order = k + 1;
                let h = match order {
                    1 | 2 => 1e-3,
                    3 => 3e-3,
                    4 => 1e-2,
                    _ => unreachable!("stencil supports derivative orders 1..=4"),
                };
                let fd = central(&log_pdf_value, eta, order, h);
                assert!(
                    (analytic - fd).abs() <= 5e-4 * analytic.abs().max(1.0) + 1e-6,
                    "logpdf d{order} mismatch for {link:?} at eta={eta}: analytic={analytic} fd={fd}"
                );
            }
        }
    }
}

#[test]
fn exact_log_pdf_derivatives_match_probit_closed_form() {
    let eta = 3.25;
    let (logf, d1, d2, d3, d4) = SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
        &InverseLink::Standard(StandardLink::Probit),
        eta,
        0.0,
    )
    .expect("exact probit log-pdf derivatives");
    let expected_logf = -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((logf - expected_logf).abs() <= 1e-15);
    assert!((d1 + eta).abs() <= 1e-15);
    assert!((d2 + 1.0).abs() <= 1e-15);
    assert_eq!(d3, 0.0);
    assert_eq!(d4, 0.0);
}

#[test]
fn exact_log_pdf_derivatives_rescaled_scale_cloglog_uniformly() {
    let eta = 501.0;
    let log_scale = 1.0;
    let (logf, d1, d2, d3, d4) = SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
        &InverseLink::Standard(StandardLink::CLogLog),
        eta,
        log_scale,
    )
    .expect("rescaled cloglog log-pdf derivatives");
    let (unscaled_logf, u1, u2, u3, u4) =
        SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
            0.0,
        )
        .expect("unscaled cloglog log-pdf derivatives");
    let scale = (-log_scale).exp();
    let expected_d1 = scale * u1;
    let expected_d2 = scale * u2;
    let expected_d3 = scale * u3;
    let expected_d4 = scale * u4;

    assert_eq!(logf, unscaled_logf);
    assert!((d1 - expected_d1).abs() <= 1e-12 * expected_d1.abs());
    assert!((d2 - expected_d2).abs() <= 1e-12 * expected_d2.abs());
    assert!((d3 - expected_d3).abs() <= 1e-12 * expected_d3.abs());
    assert!((d4 - expected_d4).abs() <= 1e-12 * expected_d4.abs());
}

#[test]
fn exact_survival_neglog_derivatives_rescaled_scale_cloglog_uniformly() {
    // The survival ratio stack must carry the SAME exp(-L) derivative rescale
    // as the log-pdf stack: the two enter the joint Hessian side by side, and
    // the logdet correction `logdet(H_exact) = logdet(H_scaled) + p*L` is only
    // valid if EVERY row's curvature (event, censored, and left-truncated
    // alike) is scaled uniformly. The log S value channel stays unshifted.
    let eta = 2.25_f64;
    let log_scale = 1.5_f64;
    let raw = eta.exp();
    let scaled = (eta - log_scale).exp();

    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
            log_scale,
        )
        .expect("rescaled cloglog survival derivatives");

    assert!((log_s + raw).abs() <= 1e-15 * raw);
    for (label, actual) in [("r", r), ("dr", dr), ("ddr", ddr), ("dddr", dddr)] {
        assert!(
            (actual - scaled).abs() <= 1e-15 * scaled,
            "CLogLog survival ratio derivative {label} must scale by exp(-L): actual={actual} expected={scaled}"
        );
    }

    let ((pair_log_s, pair_r, pair_dr, pair_ddr, pair_dddr), _) =
        SurvivalLocationScaleFamily::clglog_exit_pair(eta, log_scale);
    assert!((pair_log_s + raw).abs() <= 1e-15 * raw);
    for (label, actual) in [
        ("pair r", pair_r),
        ("pair dr", pair_dr),
        ("pair ddr", pair_ddr),
        ("pair dddr", pair_dddr),
    ] {
        assert!(
            (actual - scaled).abs() <= 1e-15 * scaled,
            "fused CLogLog survival ratio derivative {label} must scale by exp(-L): actual={actual} expected={scaled}"
        );
    }
}

#[test]
fn exact_survival_neglog_derivatives_match_identity_closed_form() {
    let eta = 0.25;
    let s = 1.0 - eta;
    let inv = 1.0 / s;
    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            &InverseLink::Standard(StandardLink::Identity),
            eta,
            0.0,
        )
        .expect("exact identity survival derivatives");
    assert!((log_s - s.ln()).abs() <= 1e-15);
    assert!((r - inv).abs() <= 1e-15);
    assert!((dr - inv * inv).abs() <= 1e-15);
    assert!((ddr - 2.0 * inv.powi(3)).abs() <= 1e-15);
    assert!((dddr - 6.0 * inv.powi(4)).abs() <= 1e-12);
}
