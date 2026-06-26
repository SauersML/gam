//! Owed-work regression gate for issue #1409 — top_k must be part of the
//! softmax SAE optimization, not a post-fit projection.
//!
//! Before the fix, an explicit `top_k` cap was applied ONLY after the full-`K`
//! Newton solve completed (a hard projection in the FFI), so:
//!   * the Newton assembly/solve still cost `O(K)` per row,
//!   * softmax still optimized all `K` atoms,
//!   * the projection did a full `O(K log K)` sort per row.
//!
//! The landed fix routes `top_k` through `set_softmax_active_cap`, which makes
//! `softmax_active_plan` engage the compact per-row active-set layout during
//! assembly. The assembled per-row block then tracks the row's top-`k` support
//! instead of full `K`.
//!
//! This test pins the OBSERVABLE optimization-level contract through the public
//! crate API only: with an explicit `top_k` cap, the per-row latent block
//! dimension is bounded by `O(top_k)` and is IDENTICAL across two widely
//! separated `K` values (independent of total `K`). A post-fit projection over
//! a full-`K` solve could not produce a `K`-independent assembly dimension.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::{Array2, Array3};

use gam::terms::{
    AssignmentMode, LatentManifold, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm,
};

/// Build a softmax SAE term whose per-row softmax mass concentrates on a planted
/// top-`k` support (large positive logits on the planted atoms, a deep floor
/// everywhere else), with a shared constant+linear 1-D Euclidean basis per atom.
/// The planted atoms are spread across the full `K` range, so a correct
/// top-`k` selection must scan all `K`, not just a prefix.
fn planted_softmax_term(
    n: usize,
    k_atoms: usize,
    planted: &[Vec<usize>],
    p: usize,
) -> (SaeManifoldTerm, Array2<f64>) {
    assert_eq!(planted.len(), n);
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 / n as f64) - 0.5);
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, 1));
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = coords[[row, 0]];
            jet[[row, 1, 0]] = 1.0;
        }
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

/// #1409: an explicit `top_k` cap must bound the SOFTMAX OPTIMIZATION, not just
/// post-process its output. With the cap set, `assemble_arrow_schur` engages the
/// compact per-row active-set layout, so each row's latent block dimension is
/// bounded by `O(top_k)` and is INDEPENDENT of total `K`. The post-fit
/// projection the issue describes (full-`K` solve, then truncate) would instead
/// produce a per-row assembly dimension that grows with `K`.
#[test]
fn top_k_bounds_softmax_assembly_dim_independent_of_total_k_1409() {
    let n = 8usize;
    let p = 4usize;
    let top_k = 3usize;
    // Two separated totals at a 2× ratio. The K-INVARIANCE of the per-row dim is
    // the contract, NOT the absolute K: `assemble_arrow_schur`'s softmax majorizer
    // setup is irreducibly O(K) (each atom's entropy-Hessian Gershgorin entry
    // couples all K assignment masses), so a large absolute K costs minutes of
    // wall-clock yet proves nothing more than a 2× ratio at a moderate K. Matches
    // the sibling #1450 gate's small-K design (`tests/owed_1450.rs`).
    const K_SMALL: usize = 512;
    const K_LARGE: usize = 1_024;
    // Planted top-`top_k` support per row, spread low/mid/high across the K range
    // (the same intended active set at BOTH K, all within K_SMALL) so a correct
    // selection must scan all K, not just a prefix.
    let planted: Vec<Vec<usize>> = (0..n)
        .map(|row| vec![row, K_SMALL / 2 + row, K_SMALL - 100 + row])
        .collect();

    // Assemble at the two separated K with the same explicit top_k cap.
    let assemble_dims = |k_atoms: usize| -> Vec<usize> {
        let (mut term, target) = planted_softmax_term(n, k_atoms, &planted, p);
        // The #1409 fix: fold top_k into the optimization.
        term.set_softmax_active_cap(Some(top_k));
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![ndarray::Array1::<f64>::zeros(1); k_atoms]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("compact softmax assembly under top_k must succeed at high K");
        // Each per-row block must be square and consistent (htt vs gt).
        for r in &sys.rows {
            assert_eq!(r.htt.nrows(), r.htt.ncols());
            assert_eq!(r.htt.nrows(), r.gt.len());
        }
        sys.rows.iter().map(|r| r.htt.nrows()).collect()
    };

    let dims_small = assemble_dims(K_SMALL);
    let dims_large = assemble_dims(K_LARGE);

    // Per-row block dim is bounded by the active contract `top_k·(1 + d)` for
    // d = 1 coord per atom, i.e. `top_k·2`. A full-`K` softmax block would be
    // `(K - 1)` free logits + `K` coord axes, i.e. ~1023 / ~2047.
    let bound = top_k * (1 + 1);
    for row in 0..n {
        assert!(
            dims_small[row] <= bound,
            "row {row}: K={K_SMALL} compact dim {} exceeds O(top_k) bound {bound} — \
             top_k did not bound the optimization (post-fit projection regression)",
            dims_small[row]
        );
        assert_eq!(
            dims_small[row], dims_large[row],
            "row {row}: per-row assembly dim must be INDEPENDENT of total K under an \
             explicit top_k cap: K={K_SMALL} gave {} but K={K_LARGE} gave {} — a post-fit \
             projection over a full-K solve would scale with K",
            dims_small[row], dims_large[row]
        );
    }

    // The compact total assembly work must be orders of magnitude below the
    // full-`K` dense block even at the smaller K_SMALL.
    let compact_work: usize = dims_small.iter().map(|&q| q * q).sum();
    let dense_q = (K_SMALL - 1) + K_SMALL; // (K-1) free logits + K coord axes
    let dense_work = n * dense_q * dense_q;
    assert!(
        compact_work * 100 < dense_work,
        "compact assembly work {compact_work} must be << full-K dense work {dense_work}"
    );
}

/// #1409 companion: at a small `K` that fits the in-core budget, setting the
/// `top_k` cap must SHRINK the per-row block relative to the uncapped solve. This
/// isolates the cap as the lever (no memory-budget engagement at small K), and
/// proves the fix actually bounds the optimization rather than being a no-op that
/// defers to a post-fit projection over the same full-`K` block.
#[test]
fn top_k_cap_shrinks_softmax_block_vs_uncapped_at_small_k_1409() {
    let n = 6usize;
    let p = 3usize;
    let k_atoms = 8usize;
    let top_k = 2usize;
    let planted: Vec<Vec<usize>> = (0..n)
        .map(|row| vec![row % k_atoms, (row + 3) % k_atoms])
        .collect();

    let dims = |cap: Option<usize>| -> Vec<usize> {
        let (mut term, target) = planted_softmax_term(n, k_atoms, &planted, p);
        term.set_softmax_active_cap(cap);
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![ndarray::Array1::<f64>::zeros(1); k_atoms]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("softmax assembly must succeed at small K");
        sys.rows.iter().map(|r| r.htt.nrows()).collect()
    };

    let uncapped = dims(None);
    let capped = dims(Some(top_k));

    // Every uncapped row block must be strictly larger than its capped
    // counterpart — the cap removed the dropped-tail logit/coord slots from the
    // assembled block, which a post-fit projection (operating after a full-K
    // solve over an unchanged block) could never do.
    let capped_bound = top_k * (1 + 1); // top_k·(|active| + Σ d_k), d_k = 1
    for row in 0..n {
        assert!(
            capped[row] <= capped_bound,
            "row {row}: capped block dim {} exceeds O(top_k) bound {capped_bound}",
            capped[row]
        );
        assert!(
            capped[row] < uncapped[row],
            "row {row}: top_k cap must SHRINK the assembled block (capped {} < uncapped {}) — \
             a no-op cap that only post-projects would leave the block at the uncapped size",
            capped[row],
            uncapped[row]
        );
    }
}
