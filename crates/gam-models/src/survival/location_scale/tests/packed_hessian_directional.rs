//! The packed-pair first directional derivative of the survival-LS joint
//! Hessian against the generic per-row pullback of the same contracted third
//! tensor, on the full data and on a Horvitz-Thompson row set.
#![cfg(test)]

use super::*;
use crate::outer_subsample::WeightedOuterRow;
use crate::row_kernel::{RowKernel, RowSet, row_kernel_directional_derivative};

/// `Σ |summand|` per entry of `Σ_i w_i Jᵢᵀ T³ᵢ[Jᵢ·d] Jᵢ` over `rows`: each row's
/// contracted third tensor pulled back one symmetric entry pair at a time, in
/// absolute value, so a cancellation inside a row does not hide its rounding.
fn summand_magnitudes(
    kernel: &SurvivalLsRowKernel<'_>,
    rows: &[(usize, f64)],
    direction: &[f64],
) -> Array2<f64> {
    let p = kernel.n_coefficients();
    let mut magnitudes = Array2::<f64>::zeros((p, p));
    for &(row, weight) in rows {
        let row_direction = kernel.jacobian_action(row, direction);
        let third = kernel
            .row_third_contracted(row, &row_direction)
            .expect("row third contraction");
        for c in 0..SLS_ROW_K {
            for d in c..SLS_ROW_K {
                if third[c][d] == 0.0 {
                    continue;
                }
                let mut single = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                single[c][d] = weight * third[c][d];
                single[d][c] = weight * third[d][c];
                let mut term = Array2::<f64>::zeros((p, p));
                kernel.add_pullback_hessian(row, &single, &mut term);
                magnitudes += &term.mapv(f64::abs);
            }
        }
    }
    magnitudes
}

fn assert_matches_generic_pullback(
    label: &str,
    family: &SurvivalLocationScaleFamily,
    states: &[ParameterBlockState],
    subsample: &[(usize, f64)],
) {
    let dynamic = family
        .build_dynamic_geometry(states)
        .expect("dynamic geometry");
    let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
    let p = kernel.n_coefficients();
    let direction = (0..p)
        .map(|j| 0.3 - 0.17 * j as f64)
        .collect::<Vec<_>>();
    let all = (0..family.n).map(|row| (row, 1.0)).collect::<Vec<_>>();
    let listed = RowSet::Subsample {
        rows: Arc::new(
            subsample
                .iter()
                .map(|&(index, weight)| WeightedOuterRow {
                    index,
                    weight,
                    stratum: 0,
                })
                .collect(),
        ),
        n_full: family.n,
    };
    for (set_label, rows, members) in [
        ("all rows", RowSet::All, all.as_slice()),
        ("weighted subsample", listed, subsample),
    ] {
        let generic = crate::row_kernel::row_kernel_directional_derivative_generic(
            &kernel, &rows, &direction,
        )
        .expect("generic per-row directional derivative");
        let packed = family
            .survival_ls_coefficient_hessian_directional_derivative(&dynamic, 0.0, &rows, &direction)
            .expect("packed directional derivative");
        // The dispatcher takes the override on every row set.
        let dispatched = row_kernel_directional_derivative(&kernel, &rows, &direction)
            .expect("dispatched directional derivative");
        assert_eq!(dispatched, packed, "{label}, {set_label}: dispatch");
        // Both sums add the same products in different orders: a row sum, each
        // row's K×K pullback, and the packed plan's pair folds.
        let growth = gam_linalg::roundoff::accumulation_growth(
            members.len() + SLS_ROW_K * SLS_ROW_K + SLS_HESSIAN_PAIRS.len(),
        );
        let band = summand_magnitudes(&kernel, members, &direction).mapv(|m| growth * m);
        assert_eq!(packed.dim(), generic.dim(), "{label}, {set_label}: shape");
        for ((a, b), &reference) in generic.indexed_iter() {
            let value = packed[[a, b]];
            assert!(
                (value - reference).abs() <= 2.0 * band[[a, b]],
                "{label}, {set_label}: [{a}][{b}] packed {value} != generic {reference} \
                 (band {})",
                band[[a, b]]
            );
        }
    }
}

/// The third tensor is live only inside each index atom's axes, the Hessian's
/// own structural pairs, so the packed lowering drops nothing, and a
/// Horvitz-Thompson row set is the same weighted sum over its own rows. Pinned
/// within the sums' measured rounding band on the fully time-varying oracle
/// family, which keeps every pair group, for three residual laws, and on the
/// time-invariant family, whose entry channels merge onto exit.
#[test]
fn packed_hessian_directional_derivative_is_the_generic_row_pullback() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let primaries: Vec<[f64; SLS_ROW_K]> = vec![
                [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
                [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
                [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
                [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, 0.25],
                [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
                [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
            ];
            let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
            let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];
            let subsample = [(0, 1.7), (2, 0.4), (3, 2.3), (5, 1.1)];
            for distribution in [
                ResidualDistribution::Gaussian,
                ResidualDistribution::Gumbel,
                ResidualDistribution::Logistic,
            ] {
                let inverse_link = residual_distribution_inverse_link(distribution);
                let family =
                    survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
                let states = survival_ls_joint_oracle_states(&primaries);
                assert_matches_generic_pullback(
                    &format!("time-varying {distribution:?}"),
                    &family,
                    &states,
                    &subsample,
                );
            }
            let family = survival_exact_newton_test_family();
            let states = survival_exact_newton_test_states(&family, 0.3, -0.4, 0.2);
            assert_matches_generic_pullback(
                "time-invariant Gaussian",
                &family,
                &states,
                &[(0, 1.5), (2, 0.6)],
            );
        })
        .expect("spawn wide-stack directional oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS packed directional oracle thread must complete"
    );
}
