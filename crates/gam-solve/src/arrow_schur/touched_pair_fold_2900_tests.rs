//! #2900 — the parallel reduced-Schur fold stores only the `(a, b)` pairs its rows
//! touch, and gives the same f64 values as the dense zero-seeded `k×k` chunk partials
//! it replaced, except for the sign of a zero entry.
//!
//! The serial in-place reduction is the control: it associates the reduction sum
//! across chunk boundaries differently, so it must NOT be bit-equal to the chunked
//! fold on this fixture, which shows the equality below can fail.
//!
//! #2822 — every partial is a dense store (value array and a touched column set per
//! left index), and the memory governor sets only how many are alive at once, down
//! to the serial in-place reduction when it admits none. The partial count does not
//! move a word, signed zeros included. A declined footprint used to fall back to a
//! keyed store that hashed every product (24.5 s against 1.2 s dense per
//! `inner_fit_core_scaling` solve, sw4i 1255679).

#![cfg(test)]

use super::*;
use crate::arrow_schur::newton_step::{
    SchurReductionKind, factor_blocks_for_system, subtract_row_schur_contribution,
};
use crate::arrow_schur::reduced_solve::{
    SCHUR_MATVEC_PARALLEL_ROW_MIN, TouchedPairFold, fold_row_chunk_partials,
    fold_touched_pair_chunk_partials, plan_touched_pair_fold, reduce_row_schur_contributions,
};
use crate::arrow_schur::solve_options::{ArrowEvidencePolicy, CpuBatchedBlockSolver};
use ndarray::{Array1, Array2, ArrayView1};
use std::cell::Cell;
use std::sync::Arc;

/// Rows with `active` of `k` border columns each, `H_tt = 4·I`, and fixed couplings
/// installed through the opaque row operator.
fn sparse_row_system(n: usize, active: usize, k: usize) -> ArrowSchurSystem {
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
    let couplings: Vec<f64> = (0..n * active * active)
        .map(|idx| 0.1 * (((idx.wrapping_mul(2_654_435_761)) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    let supports = Arc::new(supports);
    let couplings = Arc::new(couplings);
    let mut sys =
        ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(vec![active; n], k, 0);
    for row in 0..n {
        for r in 0..active {
            sys.rows[row].htt[[r, r]] = 4.0;
        }
    }
    sys.hbb = Array2::<f64>::eye(k) * 20.0;
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
fn touched_pair_fold_matches_the_dense_chunk_partials_up_to_zero_sign_2900() {
    let (n, active, k) = (1024usize, 6usize, 48usize);
    let sys = sparse_row_system(n, active, k);
    let backend = CpuBatchedBlockSolver;
    let factors = factor_blocks_for_system(
        &sys,
        0.0,
        ArrowEvidencePolicy::Strict,
        &backend,
        gam_gpu::GpuPolicy::Off,
    )
    .expect("row factors")
    .factors;
    assert!(
        n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && gam_runtime::parallel::at_top_level(),
        "the parallel chunk fold must be the route under test"
    );

    for kind in [SchurReductionKind::Direct, SchurReductionKind::SqrtBa] {
        let mut touched = Array2::<f64>::eye(k) * 20.0;
        reduce_row_schur_contributions(
            &sys,
            &factors,
            &backend,
            kind,
            &mut touched,
            gam_gpu::GpuPolicy::Off,
        )
        .expect("touched-pair chunk fold");

        // The route it replaced: one zero-seeded dense `k×k` partial per chunk,
        // folded entry by entry in chunk order.
        let mut dense = Array2::<f64>::eye(k) * 20.0;
        fold_row_chunk_partials(
            n,
            || Array2::<f64>::zeros((k, k)),
            |partial| partial.fill(0.0),
            |i, partial| {
                subtract_row_schur_contribution(
                    &sys,
                    i,
                    &sys.rows[i],
                    factors.factor(i),
                    &backend,
                    kind,
                    partial,
                )
            },
            |partial| {
                for a in 0..k {
                    for b in 0..k {
                        dense[[a, b]] += partial[[a, b]];
                    }
                }
            },
        )
        .expect("dense chunk fold");

        let mut serial = Array2::<f64>::eye(k) * 20.0;
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool")
            .install(|| {
                reduce_row_schur_contributions(
                    &sys,
                    &factors,
                    &backend,
                    kind,
                    &mut serial,
                    gam_gpu::GpuPolicy::Off,
                )
            })
            .expect("serial in-place reduction");

        assert!(
            bit_equal_up_to_zero_sign(&touched, &dense),
            "{kind:?}: the touched-pair fold departs from the dense chunk partials"
        );
        assert!(
            !bit_equal_up_to_zero_sign(&serial, &dense),
            "{kind:?}: control failed: the serial reduction is bit-equal to the chunked fold, \
             so bit equality cannot tell the two routes apart on this fixture"
        );
    }
}

fn raw_words_equal(left: &Array2<f64>, right: &Array2<f64>) -> bool {
    left.dim() == right.dim()
        && left
            .iter()
            .zip(right.iter())
            .all(|(p, q)| p.to_bits() == q.to_bits())
}

/// The production fold at the governor-admitted width, and at one and two live
/// partials, against an independent touched-pairs-only reference: zero-seeded dense
/// `k×k` chunk partials under the CPU `block_gemm_subtract`, folded in chunk order
/// at every entry whose word is not `+0.0`.
///
/// That reference folds exactly the touched pairs' words. A pair no row touches is
/// `+0.0` in its partial and skipped. A touched pair whose sum is `+0.0` is skipped
/// too, where the production fold adds it, but the Schur seed `20·I` holds `+0.0`
/// off its diagonal and a sum of `±0.0` terms onto `+0.0` stays `+0.0`, so neither
/// choice moves a word on this fixture.
///
/// `k = 48` holds each touched set in one word; `k = 130` spans three, the last one
/// partial.
#[test]
fn the_touched_pair_fold_folds_the_same_words_at_every_partial_count_2822() {
    for k in [48usize, 130] {
        assert_touched_pair_fold_words(k);
    }
}

fn assert_touched_pair_fold_words(k: usize) {
    let (n, active) = (1024usize, 6usize);
    let sys = sparse_row_system(n, active, k);
    let backend = CpuBatchedBlockSolver;
    let factors = factor_blocks_for_system(
        &sys,
        0.0,
        ArrowEvidencePolicy::Strict,
        &backend,
        gam_gpu::GpuPolicy::Off,
    )
    .expect("row factors")
    .factors;
    assert!(
        n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && gam_runtime::parallel::at_top_level(),
        "the parallel chunk fold must be the route under test"
    );

    for kind in [SchurReductionKind::Direct, SchurReductionKind::SqrtBa] {
        let mut admitted = Array2::<f64>::eye(k) * 20.0;
        reduce_row_schur_contributions(
            &sys,
            &factors,
            &backend,
            kind,
            &mut admitted,
            gam_gpu::GpuPolicy::Off,
        )
        .expect("touched-pair chunk fold at the admitted width");
        let at_width = |partials: usize| {
            let mut schur = Array2::<f64>::eye(k) * 20.0;
            fold_touched_pair_chunk_partials(&sys, &factors, &backend, kind, &mut schur, partials)
                .expect("touched-pair chunk fold");
            schur
        };

        let mut reference = Array2::<f64>::eye(k) * 20.0;
        fold_row_chunk_partials(
            n,
            || Array2::<f64>::zeros((k, k)),
            |partial| partial.fill(0.0),
            |i, partial| {
                subtract_row_schur_contribution(
                    &sys,
                    i,
                    &sys.rows[i],
                    factors.factor(i),
                    &backend,
                    kind,
                    partial,
                )
            },
            |partial| {
                for ((a, b), value) in partial.indexed_iter() {
                    if value.to_bits() != 0 {
                        reference[[a, b]] += *value;
                    }
                }
            },
        )
        .expect("reference chunk fold");

        let mut serial = Array2::<f64>::eye(k) * 20.0;
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool")
            .install(|| {
                reduce_row_schur_contributions(
                    &sys,
                    &factors,
                    &backend,
                    kind,
                    &mut serial,
                    gam_gpu::GpuPolicy::Off,
                )
            })
            .expect("serial in-place reduction");

        assert!(
            raw_words_equal(&admitted, &reference),
            "{kind:?} at k = {k}: the touched-pair fold at the admitted width folds different \
             words from the touched-pairs-only reference"
        );
        for partials in [1, 2] {
            assert!(
                raw_words_equal(&at_width(partials), &admitted),
                "{kind:?} at k = {k}: the touched-pair fold at {partials} live partials folds \
                 different words from the fold at the admitted width"
            );
        }
        assert!(
            !raw_words_equal(&serial, &admitted),
            "{kind:?} at k = {k}: control failed: the serial reduction's words equal the chunked \
             fold's, so word equality cannot tell two reductions apart on this fixture"
        );
    }
}

/// One dense partial's charge at border `k`: `k²` values, `k·⌈k/64⌉` touched words
/// and a list slot and flag per left index.
fn dense_partial_bytes(k: usize) -> usize {
    k * k * 8 + k * k.div_ceil(64) * 8 + k * (8 + 1)
}

/// A declined footprint costs the fold its parallelism, never its per-product work.
/// The plan admits as many dense partials as the ledger takes, down to the serial
/// in-place reduction. Any other fallback fails the exact `Chunked { partials: 2 }`
/// below, whether it is a hashed store, the in-place loop taken too early or a width
/// the ledger did not admit. The ledger here is a budget the test sets: `remaining`
/// reports it and `reserve` charges it.
#[test]
fn a_declined_footprint_folds_with_fewer_dense_partials_never_a_hashed_store_2822() {
    let k = 48usize;
    let per_partial = dense_partial_bytes(k);

    // The budget holds two of the four partials the pool wants.
    let budget = 2 * per_partial + per_partial / 2;
    let reserved = Cell::new(0usize);
    let (plan, charge) = plan_touched_pair_fold(
        k,
        4,
        || budget - reserved.get(),
        |bytes| {
            (reserved.get() + bytes <= budget).then(|| {
                reserved.set(reserved.get() + bytes);
                bytes
            })
        },
    );
    assert_eq!(
        plan,
        TouchedPairFold::Chunked { partials: 2 },
        "a budget of {budget} bytes admits two {per_partial}-byte dense partials"
    );
    assert_eq!(
        charge,
        Some(2 * per_partial),
        "the admitted partials are charged as dense stores"
    );

    // The whole want fits: every partial is admitted.
    let (plan, _) = plan_touched_pair_fold(k, 4, || 4 * per_partial, Some);
    assert_eq!(plan, TouchedPairFold::Chunked { partials: 4 });

    // A peer reserves between the price and the charge: the count that remains is taken.
    let peer_took = Cell::new(false);
    let (plan, charge) = plan_touched_pair_fold(
        k,
        4,
        || {
            if peer_took.get() {
                per_partial
            } else {
                3 * per_partial
            }
        },
        |bytes| {
            if peer_took.get() {
                Some(bytes)
            } else {
                peer_took.set(true);
                None
            }
        },
    );
    assert_eq!(plan, TouchedPairFold::Chunked { partials: 1 });
    assert_eq!(charge, Some(per_partial));

    // Less than one partial, or a byte count that overflows: the rows reduce in place.
    let (plan, charge) = plan_touched_pair_fold(k, 4, || per_partial - 1, Some);
    assert_eq!((plan, charge), (TouchedPairFold::InPlace, None));
    let (plan, charge) = plan_touched_pair_fold(usize::MAX, 4, || usize::MAX, Some);
    assert_eq!((plan, charge), (TouchedPairFold::InPlace, None));
}
