//! #2900 — `solve_arrow_newton_step_with_options` carries the Direct step's dense
//! reduced Schur forward as the evidence Schur when the step ran at ridge zero, no
//! step row was ridge-escalated and the evidence factorization deflated nothing, and
//! the evidence factor it then produces equals the one assembled from scratch.
//!
//! The control is a row whose smallest pivot is below `safe_spd_pivot_min`: the
//! Strict step factorization must ridge-escalate it while the evidence factorization
//! keeps the raw factor, so the two row slabs differ, the step's Schur is not the
//! evidence Schur, and the cache must still equal the from-scratch evidence factor.

#![cfg(test)]

use super::*;
use crate::arrow_schur::newton_step::{
    factor_blocks_for_system, solve_arrow_newton_step_artifacts,
};
use crate::arrow_schur::reduced_solve::{
    build_dense_schur_direct, exact_a_reduced_classification,
    factor_dense_reduced_schur_with_exact_a,
};
use crate::arrow_schur::solve_options::CpuBatchedBlockSolver;
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

/// Rows with `active` of `k` border columns, `H_tt = 4·I`, fixed couplings through
/// the opaque row operator. With `weak_first_coordinate`, row 0's first latent
/// coordinate has pivot `1e-9` and no coupling to the border, so it changes the
/// row factorization without entering the reduced Schur through that coordinate.
fn coupled_row_system(
    n: usize,
    active: usize,
    k: usize,
    weak_first_coordinate: bool,
) -> ArrowSchurSystem {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut mark = vec![usize::MAX; k];
    let mut supports: Vec<u32> = Vec::with_capacity(n * active);
    for row in 0..n {
        let first = supports.len();
        while supports.len() - first < active {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let atom = ((state >> 33) % k as u64) as usize;
            if mark[atom] != row {
                mark[atom] = row;
                supports.push(atom as u32);
            }
        }
        supports[first..].sort_unstable();
    }
    let mut couplings: Vec<f64> = (0..n * active * active)
        .map(|idx| 0.1 * (((idx.wrapping_mul(2_654_435_761)) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    let mut sys =
        ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(vec![active; n], k, 0);
    for row in 0..n {
        for r in 0..active {
            sys.rows[row].htt[[r, r]] = 4.0;
        }
    }
    if weak_first_coordinate {
        sys.rows[0].htt[[0, 0]] = 1e-9;
        for coupling in couplings.iter_mut().take(active) {
            *coupling = 0.0;
        }
    }
    sys.hbb = Array2::<f64>::eye(k) * 20.0;
    let supports = Arc::new(supports);
    let couplings = Arc::new(couplings);
    let (forward_supports, forward_couplings) = (Arc::clone(&supports), Arc::clone(&couplings));
    let (transpose_supports, transpose_couplings) = (Arc::clone(&supports), Arc::clone(&couplings));
    // Each row's supports are distinct, so its couplings are exactly the row's entries.
    let row_norm_bounds: Arc<[f64]> = couplings
        .chunks(active * active)
        .map(|row_couplings| frobenius_norm_upper_bound(row_couplings.iter().copied()))
        .collect();
    sys.set_row_htbeta_operator(
        move |row: usize, x: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            for r in 0..active {
                let mut acc = 0.0_f64;
                for ci in 0..active {
                    acc += forward_couplings[(row * active + r) * active + ci]
                        * x[forward_supports[row * active + ci] as usize];
                }
                out[r] += acc;
            }
        },
        move |row: usize, v: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            for ci in 0..active {
                let mut acc = 0.0_f64;
                for r in 0..active {
                    acc += transpose_couplings[(row * active + r) * active + ci] * v[r];
                }
                out[transpose_supports[row * active + ci] as usize] += acc;
            }
        },
        // Each apply accumulates `active` terms, one more for the transpose's addition.
        RowHtbetaDeclaration {
            row_norm_bounds,
            apply_depth: active + 1,
        },
    );
    sys
}

fn bit_equal_up_to_zero_sign(left: &Array2<f64>, right: &Array2<f64>) -> bool {
    left.dim() == right.dim() && left.iter().zip(right.iter()).all(|(p, q)| p == q)
}

#[test]
fn evidence_cache_carries_the_step_schur_when_the_row_factors_coincide_2900() {
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let backend = CpuBatchedBlockSolver;
    for weak_first_coordinate in [false, true] {
        let sys = coupled_row_system(300, 6, 48, weak_first_coordinate);

        // From scratch: undamped evidence row factors → dense Schur → evidence factor.
        // The fixture installs no β-gauge quotient, so the evidence pin is the identity.
        let undamped = factor_blocks_for_system(
            &sys,
            0.0,
            options.evidence_policy,
            &backend,
            options.gpu_policy,
        )
        .expect("undamped evidence row factors");
        let evidence_schur =
            build_dense_schur_direct(&sys, &undamped.factors, 0.0, &backend, options.gpu_policy)
                .expect("from-scratch evidence Schur");
        let classification = exact_a_reduced_classification(&sys, &undamped.factors)
            .expect("exact-A reduced classification");
        let evidence_factor = factor_dense_reduced_schur_with_exact_a(
            &evidence_schur,
            options.evidence_policy.reduced_schur_policy(),
            classification.as_ref(),
        )
        .expect("from-scratch evidence Schur factor")
        .factor;

        let artifacts = solve_arrow_newton_step_artifacts(&sys, 0.0, 0.0, &options)
            .expect("Direct step artifacts");
        let step_schur = artifacts
            .step_schur
            .as_ref()
            .expect("the Direct step keeps its Schur by value");
        let step_escalations = artifacts.step_ridge_escalated_rows;

        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("Direct step with evidence cache");
        let cached_factor = cache
            .schur_factor
            .as_ref()
            .expect("evidence Schur factor");
        assert!(
            bit_equal_up_to_zero_sign(cached_factor, &evidence_factor),
            "weak_first_coordinate={weak_first_coordinate}: the cached evidence factor departs \
             from the from-scratch evidence build"
        );
        if weak_first_coordinate {
            assert!(
                matches!(step_escalations, Some(count) if count > 0),
                "control: the weak pivot must make the Strict step ridge-escalate row 0; \
                 escalations reported {step_escalations:?}"
            );
            assert!(
                !bit_equal_up_to_zero_sign(step_schur, &evidence_schur),
                "control: the escalated step's Schur must differ from the evidence Schur, or \
                 reusing it could not be told apart from rebuilding"
            );
        } else {
            assert!(
                step_escalations == Some(0),
                "the benign fixture must factor every step row at the base ridge; escalations \
                 reported {step_escalations:?}"
            );
            assert!(
                bit_equal_up_to_zero_sign(step_schur, &evidence_schur),
                "the carried step Schur must be the evidence Schur"
            );
        }
    }
}
