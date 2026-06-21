//! Owed-work regression gate for issue #1408.
//!
//! #1408 — "Softmax assignment never uses advertised compact high-`K` row
//! layout." The dispatch in `assemble_arrow_schur` used to read
//! `AssignmentMode::Softmax { .. } => None`, so a softmax SAE always retained the
//! full `K`-atom per-row block and the documented compact top-`k` layout was a
//! dead contract; any `top_k` was applied only as an after-the-fit projection.
//!
//! Fixed by commit 9676d5556, which replaced that `=> None` arm with a real
//! dispatch: softmax now consults `softmax_active_plan()` (driven by the user's
//! `top_k` via `set_softmax_active_cap`) and, on `Some`, builds the compact
//! active-set layout via `SaeRowLayout::from_dense_weights` — folding top-`k`
//! sparsity INTO the optimization, exactly like the IBP-MAP branch.
//!
//! This test pins the externally observable consequence through the PUBLIC crate
//! API only: with an explicit `top_k` cap, the assembled Arrow-Schur per-row
//! block dimension is compact (bounded by `O(top_k)`) and INDEPENDENT of the
//! total `K`, instead of growing with `K` as the buggy full-`K` path did.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::{Array1, Array2, Array3};

use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};

/// Build a softmax SAE term whose per-row logits put large mass on a planted
/// top-`k` support (spread across the full `K` range) and a small uniform floor
/// elsewhere, so the softmax assignment concentrates on the planted atoms while
/// the dropped tail carries negligible `O(a)` mass — the regime the compact
/// softmax layout (#1408) optimizes.
fn planted_softmax_sae_term(
    n: usize,
    k_atoms: usize,
    planted: &[Vec<usize>],
    p: usize,
) -> (SaeManifoldTerm, Array2<f64>) {
    assert_eq!(planted.len(), n);
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    // Shared constant+linear basis: column 0 = 1, column 1 = t (1-D coord).
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 / n as f64) - 0.5);
    for atom_idx in 0..k_atoms {
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, 1));
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = coords[[row, 0]];
            jet[[row, 1, 0]] = 1.0;
        }
        // Distinct decoder direction per atom onto output channel `atom_idx % p`.
        let mut decoder = Array2::<f64>::zeros((2, p));
        decoder[[1, atom_idx % p]] = 0.1 + 0.01 * ((atom_idx % 7) as f64);
        atoms.push(
            SaeManifoldAtom::new(
                format!("atom{atom_idx}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(2),
            )
            .unwrap(),
        );
        coord_blocks.push(coords.clone());
        manifolds.push(LatentManifold::Euclidean);
    }
    // Logits: small uniform floor everywhere, large mass on the planted atoms.
    let mut logits = Array2::<f64>::from_elem((n, k_atoms), -6.0);
    for (row, active) in planted.iter().enumerate() {
        for &k in active {
            logits[[row, k]] = 6.0;
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, c)| 0.05 * ((row + c) as f64).sin());
    (term, target)
}

/// #1408: a `top_k`-capped softmax SAE must engage the COMPACT per-row layout
/// inside the assembly, so the assembled Arrow-Schur per-row block dimension is
/// bounded by `O(top_k)` and INDEPENDENT of the total `K`. The pre-fix dispatch
/// (`AssignmentMode::Softmax { .. } => None`) kept the full-`K` block, whose
/// dimension `q = (K-1) free logits + K coord axes` grows with `K`; this test
/// would then fail both the `O(top_k)` bound and the K-independence equality.
#[test]
fn softmax_topk_engages_compact_per_row_layout_in_assembly_1408() {
    let n = 8usize;
    let p = 4usize;
    let top_k = 3usize;
    // Each row's planted support spread across the full K range so a correct
    // selection must scan all K (not just a low-index prefix).
    let planted: Vec<Vec<usize>> = (0..n).map(|row| vec![row, 300 + row, 700 + row]).collect();

    // Assemble at two widely-separated K with the SAME top_k and planted support;
    // the public per-row block dims must be identical (independent of K) and
    // bounded by O(top_k).
    let assemble_dims = |k_atoms: usize| -> Vec<usize> {
        let (mut term, target) = planted_softmax_sae_term(n, k_atoms, &planted, p);
        // Fold top_k into the OPTIMIZATION — the #1408/#1409 fix.
        term.set_softmax_active_cap(Some(top_k));
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("compact softmax assembly must succeed at large K");
        // Each per-row block must be square and consistent with its gradient.
        for r in &sys.rows {
            assert_eq!(r.htt.nrows(), r.htt.ncols());
            assert_eq!(r.htt.nrows(), r.gt.len());
        }
        sys.rows.iter().map(|r| r.htt.nrows()).collect()
    };

    let dims_1k = assemble_dims(1_000);
    let dims_10k = assemble_dims(10_000);

    // (a) O(top_k) bound: per-row block dim is at most `top_k·(1 + d)` for d=1
    // coord (= |active| logit slots + Σ active coord axes). A full-K softmax
    // block would be ~ (K-1) + K = ~2000 at K=1000 — far above this bound.
    let bound = top_k * (1 + 1);
    for row in 0..n {
        assert!(
            dims_1k[row] <= bound,
            "row {row} K=1000 compact per-row dim {} exceeds O(top_k) bound {bound} \
             — softmax did not engage the compact layout (#1408 regression)",
            dims_1k[row]
        );
    }

    // (b) K-independence: identical per-row dims at K=1000 and K=10000. The buggy
    // full-K path would give ~2000 vs ~20000.
    for row in 0..n {
        assert_eq!(
            dims_1k[row], dims_10k[row],
            "row {row} per-row dim must be INDEPENDENT of total K (#1408 compact contract): \
             K=1000 gave {} but K=10000 gave {}",
            dims_1k[row], dims_10k[row]
        );
    }

    // (c) Quantitative: total compact work is < 1/100 of the full-K dense block
    // even at the smaller K=1000.
    let compact_work: usize = dims_1k.iter().map(|&q| q * q).sum();
    let dense_q = (1_000 - 1) + 1_000; // (K-1) free logits + K coord axes
    let dense_work = n * dense_q * dense_q;
    assert!(
        compact_work * 100 < dense_work,
        "compact work {compact_work} must be << full-K dense work {dense_work} (#1408)"
    );
}

/// #1408 control: WITHOUT a `top_k` cap and at a small in-budget `K`, the
/// softmax layout stays full-`K` (the plan returns `None`), so the assembled
/// per-row block carries every atom. This proves the compact layout in the test
/// above is genuinely DRIVEN by the cap rather than always-on, and that the cap
/// is the lever that folds top-`k` into the optimization.
#[test]
fn softmax_without_cap_at_small_k_keeps_full_row_block_1408() {
    let n = 6usize;
    let p = 3usize;
    let k_atoms = 5usize;
    let planted: Vec<Vec<usize>> = (0..n).map(|row| vec![row % k_atoms]).collect();
    let (mut term, target) = planted_softmax_sae_term(n, k_atoms, &planted, p);
    // No cap set; small K is within the in-core budget, so no compaction lever
    // engages and the dense full-K layout is retained.
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("dense softmax assembly must succeed");
    // Full-K softmax row block dim = (K-1) free logits + K coord axes (d=1).
    let full_q = (k_atoms - 1) + k_atoms;
    for r in &sys.rows {
        assert_eq!(r.htt.nrows(), r.htt.ncols());
        assert_eq!(
            r.htt.nrows(),
            full_q,
            "uncapped small-K softmax must keep the full-K per-row block dim {full_q}"
        );
    }
}
